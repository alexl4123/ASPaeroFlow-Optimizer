import zmq
import time
import argparse
import sys
import subprocess
import threading

from pathlib import Path
from typing import Any, Final, List, Optional, Callable

import base64
import json

bind_host_name = "127.0.0.1"
optimizer_connection_hostname = "127.0.0.1"
asp_aero_flow_path = "../01_ASPaeroFlow/main.py"

def start_telemetry_broker():
    """Dedicated background thread for XPUB/XSUB proxying."""
    # Isolated ZeroMQ context for the thread
    context = zmq.Context()

    global bind_host_name
    global optimizer_connection_hostname

    frontend_xsub = context.socket(zmq.XSUB)
    frontend_xsub.bind(f"tcp://{bind_host_name}:5556") 
    
    backend_xpub = context.socket(zmq.XPUB)
    backend_xpub.bind(f"tcp://{bind_host_name}:5557") 
    
    # Blocks indefinitely to handle multiplexing
    zmq.proxy(frontend_xsub, backend_xpub)


def main(argv: Optional[List[str]] = None):

    parser = argparse.ArgumentParser(
        prog="Optimization Controller",
        description="Controls certain behaviors of the optimizer, like what data is loaded."
    )

    parser.add_argument(
            "--controller-folder", help="Selectable folders for optimizer.", type=Path, default=None, metavar="FOLDER")


    parser.add_argument(
            "--default-hostname", help="Default hostname is 127.0.0.1", type=str, default="127.0.0.1")



    args = parser.parse_args(argv)

    global bind_host_name
    global optimizer_connection_hostname
    global asp_aero_flow_path

    bind_host_name = args.default_hostname


    if bind_host_name == "0.0.0.0":
        optimizer_connection_hostname = "optimizer"
        asp_aero_flow_path = "01_ASPaeroFlow/main.py"
    else:
        optimizer_connection_hostname = "127.0.0.1"

    print("[DEBUG] - Start")

    # 0. Broker Initialization
    # Spawn proxy thread before any subscribers attempt to connect
    broker_thread = threading.Thread(target=start_telemetry_broker, daemon=True)
    broker_thread.start()
    print("[DEBUG] - Telemetry broker thread spawned (XSUB:5556 -> XPUB:5557)")

    context = zmq.Context()
    print("[DEBUG] - Startup initialized")    
    
    # 1. Control Channel (PAIR)
    ctrl_socket = context.socket(zmq.PAIR)
    ctrl_socket.bind(f"tcp://{bind_host_name}:5555")
    print("[DEBUG] - Optimizer control socket create")    
    
    # 2. Telemetry Channel (SUB)
    sub_socket = context.socket(zmq.SUB)
    # UPDATED: Connect to XPUB port (5557), not XSUB
    sub_socket.connect(f"tcp://127.0.0.1:5557")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics
    print("[DEBUG] - Optimizer data socket create")



    # 1. Control Channel (PAIR)
    clinguin_ctrl_socket = context.socket(zmq.PAIR)
    clinguin_ctrl_socket.bind(f"tcp://{bind_host_name}:5570")
    print("[DEBUG] - Clinguin control socket create")    

    # Non-blocking I/O setup for the Control Channel
    clinguin_ctrl_poller = zmq.Poller()
    clinguin_ctrl_poller.register(clinguin_ctrl_socket, zmq.POLLIN)

    # 2. Telemetry Channel (SUB)
    clinguin_sub_socket = context.socket(zmq.PUB)
    clinguin_sub_socket.bind(f"tcp://{bind_host_name}:5571")
    print("[DEBUG] - Clinguin data socket create")    
    #clinguin_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics
    
    # Mitigate Slow Joiner Syndrome (allow TCP handshake to complete)
    time.sleep(1) 
    
    ######################## INITIALIZE ############################
    # Initialize state tracking variables
    optimizer_ready = False
    clinguin_ready = False

    # Configure the Poller for I/O multiplexing
    init_poller = zmq.Poller()
    init_poller.register(ctrl_socket, zmq.POLLIN)
    init_poller.register(clinguin_ctrl_socket, zmq.POLLIN)

    # Loop until both components are fully initialized
    while not (optimizer_ready and clinguin_ready):
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
            message = ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "INITIALIZED OPTIMIZER":
                print("[DEBUG] OPTIMIZER INITIALIZATION RECEIVED")
                optimizer_ready = True
                init_poller.unregister(ctrl_socket)
            else:
                print(f"[DEBUG] OPTIMIZER BUSY:\n{message}")

        # Process Clinguin control socket events
        if clinguin_ctrl_socket in socks and socks[clinguin_ctrl_socket] == zmq.POLLIN:
            message = clinguin_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "INITIALIZED CLINGUIN":
                print("[DEBUG] CLINGUIN INITIALIZATION RECEIVED")
                clinguin_ready = True
                init_poller.unregister(clinguin_ctrl_socket)
            else:
                print(f"[DEBUG] CLINGUIN BUSY:\n{message}")

    init_poller = zmq.Poller()
    init_poller.register(ctrl_socket, zmq.POLLIN)

    clinguin_init_poller = zmq.Poller()
    clinguin_init_poller.register(clinguin_ctrl_socket, zmq.POLLIN)

    if args.controller_folder is not None:
        print("[DEBUG] CONTROLLER DEFINED INSTANCE")
        ctrl_socket.send_string("CONTROLLER DEFINED INSTANCE")

        while True:
            socks = dict(init_poller.poll(1000))
            if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
                message = ctrl_socket.recv_string(flags=zmq.NOBLOCK)

                if message == "ack":
                    break
                else:
                    print(f"[DEBUG] OPTIMIZER BUSY:\n{message}")

        folders = [f for f in Path(args.controller_folder).iterdir() if f.is_dir()]

        folders_strings = []
        for folder in folders:
            folders_strings.append(str(folder))
        
        clinguin_ctrl_socket.send_string(json.dumps(folders_strings))

    else:
        print("[DEBUG] OPTIMIZER DEFINED INSTANCE")
        ctrl_socket.send_string("OPTIMIZER DEFINED INSTANCE")
        clinguin_ctrl_socket.send_string("INSTANCE LOADED")

        while True:
            socks = dict(init_poller.poll(1000))
            if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
                message = ctrl_socket.recv_string(flags=zmq.NOBLOCK)

                if message == "ack":
                    break
                else:
                    print(f"[DEBUG] OPTIMIZER BUSY:\n{message}")

    while True:
        socks = dict(clinguin_init_poller.poll(1000))
        if clinguin_ctrl_socket in socks and socks[clinguin_ctrl_socket] == zmq.POLLIN:
            message = clinguin_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "ack":
                break
            else:
                print(f"[DEBUG] CLINGUIN BUSY:\n{message}")
    
    init_poller = zmq.Poller()
    init_poller.register(ctrl_socket, zmq.POLLIN)
    ctrl_socket.send_string("GET OPTIONS")

    while True:
        socks = dict(init_poller.poll(1000))
        if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
            options = ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            break

    ctrl_socket.send_string("ack")
    init_poller = zmq.Poller()
    init_poller.register(clinguin_ctrl_socket, zmq.POLLIN)
    clinguin_ctrl_socket.send_string(options)

    while True:
        socks = dict(init_poller.poll(1000))
        if clinguin_ctrl_socket in socks and socks[clinguin_ctrl_socket] == zmq.POLLIN:
            message = clinguin_ctrl_socket.recv_string(flags=zmq.NOBLOCK)

            if message == "ack":
                break

    init_poller = zmq.Poller()
    init_poller.register(ctrl_socket, zmq.POLLIN)
    ctrl_socket.send_string("GET OBJECTIVE FUNCTIONS")

    while True:
        socks = dict(init_poller.poll(1000))
        if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
            options = ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            break

    ctrl_socket.send_string("ack")
    init_poller = zmq.Poller()
    init_poller.register(clinguin_ctrl_socket, zmq.POLLIN)
    clinguin_ctrl_socket.send_string(options)

    while True:
        socks = dict(init_poller.poll(1000))
        if clinguin_ctrl_socket in socks and socks[clinguin_ctrl_socket] == zmq.POLLIN:
            message = clinguin_ctrl_socket.recv_string(flags=zmq.NOBLOCK)

            if message == "ack":
                break


    #
    ################## OPTIMIZATION LOOP ###################
    #
    try:

        #clinguin_pub_poller = zmq.Poller()
        #clinguin_pub_poller.register(clinguin_sub_socket, zmq.POLLIN)
        # Read remaining telemetry
        paused = True
        while True:
            time.sleep(0.05) 
            #events = dict(clinguin_pub_poller.poll(timeout=0))
            #if clinguin_sub_socket in events:
            #
            computation_finished = False
            while True:
                try:
                    # Attempt non-blocking read
                    message = sub_socket.recv_string(flags=zmq.NOBLOCK)
                    print(f"[Controller] {message}")
                    clinguin_sub_socket.send_string(message)
                    #if "\"COMPUTATION-FINISHED\": true" in message:
                    #    computation_finished = True
                except zmq.Again:
                    # Queue is empty; break loop to continue computation
                    break

            if computation_finished is True:
                break

            # A. State Machine for Interrupts
            events = dict(clinguin_ctrl_poller.poll(timeout=0))
            """
            if paused is True:
                events = dict(clinguin_ctrl_poller.poll(timeout=None))
            else:
                events = dict(clinguin_ctrl_poller.poll(timeout=0))
            """

            if clinguin_ctrl_socket in events:
                command = clinguin_ctrl_socket.recv_string()
                if command == "PAUSE":
                    paused = True
                    print("[Clinguin->Controller->Optimizer] PAUSE")
                    query_poller = zmq.Poller()
                    query_poller.register(ctrl_socket, zmq.POLLIN)
                    ctrl_socket.send_string("PAUSE")

                    while True:
                        socks = dict(query_poller.poll(1000))
                        if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
                            message = ctrl_socket.recv_string(flags=zmq.NOBLOCK)
                            if message != "TELEMETRY: [STATUS] PAUSED":
                                print(">[WARN]< Optimizer response after pause does not match protocol.")
                            break
                elif command == "START":
                    paused = False
                    print("[Clinguin->Controller->Optimizer] START")


                    query_poller = zmq.Poller()
                    query_poller.register(ctrl_socket, zmq.POLLIN)
                    ctrl_socket.send_string("START")

                    while True:
                        socks = dict(query_poller.poll(1000))
                        if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
                            message = ctrl_socket.recv_string(flags=zmq.NOBLOCK)
                            if message != "TELEMETRY: [STATUS] RESUMED":
                                print(">[WARN]< Optimizer response after start does not match protocol.")
                            break

                elif command.startswith("<LOAD>"):
                    paused = True
                    command = command[6:]
                    print(command)

                    if command in folders_strings:
                        selected_folder = command
                        graph_poller = zmq.Poller()
                        graph_poller.register(ctrl_socket, zmq.POLLIN)
                        ctrl_socket.send_string(f"<LOAD>{selected_folder}")

                        while True:
                            socks = dict(graph_poller.poll(1000))
                            if ctrl_socket in socks and socks[ctrl_socket] == zmq.POLLIN:
                                message = ctrl_socket.recv_string(flags=zmq.NOBLOCK)
                                clinguin_ctrl_socket.send_string(message)
                                break

                elif command.startswith("<OPTION>"):
                    print(command)
                    ctrl_socket.send_string(command)
                elif command.startswith("<EXPLAIN>"):
                    print("<EXPLAIN>")

                    if bind_host_name == "0.0.0.0":
                        controller_hostname = "controller"
                        controller_hostname = "127.0.0.1"
                        _encoding_path = "01_ASPaeroFlow/encoding.lp"
                    else:
                        controller_hostname = "127.0.0.1"
                        _encoding_path = "../01_ASPaeroFlow/encoding.lp"

                    # 1. Rename variable to prevent shadowing the actual payload string
                    exec_args = [sys.executable, f"{asp_aero_flow_path}", "--explainability-enabled=True", f"--encoding-path={_encoding_path}", f"--controller-hostname={controller_hostname}"]



                    process = subprocess.Popen(exec_args)
                    print("[DEBUG] - Explainability process started")

                    explain_ctrl_socket = context.socket(zmq.PAIR)
                    #explain_ctrl_socket.connect(f"tcp://{optimizer_connection_hostname}:6000")
                    explain_ctrl_socket.connect(f"tcp://127.0.0.1:6000")
                    print("[DEBUG] - Explain control socket create")   

                    explain_poller = zmq.Poller()
                    explain_poller.register(explain_ctrl_socket, zmq.POLLIN)
                    print("[DEBUG] - Poller created")   

                    # 2. Handle ZMQ_PAIR mute state race condition
                    while True:
                        try:
                            # Assuming 'command' holds the original string payload (e.g., "EXPLAIN: {...}")
                            explain_ctrl_socket.send_string(command, flags=zmq.NOBLOCK)
                            print("[DEBUG] - Explain command sent")    
                            break
                        except zmq.error.Again:
                            # Peer has not bound yet. Backoff and retry.
                            time.sleep(0.05) 

                    # 3. Wait for acknowledgment
                    while True:
                        socks = dict(explain_poller.poll(1000))
                        if explain_ctrl_socket in socks and socks[explain_ctrl_socket] == zmq.POLLIN:
                            message = explain_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
                            print(f"[DEBUG] - Received XAI Ack: {message}")
                            break

                    explain_poller.unregister(explain_ctrl_socket)
                    explain_ctrl_socket.setsockopt(zmq.LINGER, 0)
                    explain_ctrl_socket.close()
                    print("[DEBUG] - Explain control socket closed")

                else:
                    print(f"[DEBUG] CLINGUIN BUSY:\n{command}")
            
            
            """
            # B. Blocking Wait Loop (halts heuristic progression)
            if paused:
                # poller.poll() with None blocks indefinitely until I/O occurs
                events = dict(clinguin_ctrl_poller.poll(timeout=None))
                if clinguin_ctrl_socket in events:
                    message = clinguin_ctrl_socket.recv_string()
                    if message == "START":
                        print("[Clinguin->Controller->Optimizer] START")
                        ctrl_socket.send_string("START")
                        paused = False
                    elif message.startswith("<LOAD>"):
                        paused = True
                        if message.startswith("<LOAD>"):
                            message = message[6:]
                            print(message)

                        if message in folders_strings:
                            selected_folder = message
                            ctrl_socket.send_string(f"<LOAD>{selected_folder}")
                    elif message.startswith("<OPTION>"):
                        ctrl_socket.send_string(message)
                        print(message)
                    else:
                        print(f"[DEBUG] CLINGUIN BUSY:\n{message}")

            if paused is True:
                continue
            """

                
    finally:
        # Graceful Context Termination
        ctrl_socket.setsockopt(zmq.LINGER, 0)
        sub_socket.setsockopt(zmq.LINGER, 0)
        ctrl_socket.close()
        sub_socket.close()
        context.term()

if __name__ == "__main__":
    main()

