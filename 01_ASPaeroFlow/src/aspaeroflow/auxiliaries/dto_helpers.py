
def convert_dto_to_global_vars(self, global_var_dto):
    target_vars = vars(self)
    for key, value in global_var_dto.items():
        target_vars[key] = value

def convert_global_vars_to_dto(self):
    global_vars = vars(self)
    return global_vars
