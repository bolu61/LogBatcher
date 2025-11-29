
import re
from logbatcher.matching import extract_variables

def vars_update(refer_log, template, candidates):
    new_variables = extract_variables(refer_log, template)
    extend_vars = []
    if not new_variables:
        return extend_vars
    for var in new_variables:
        var = re.sub(r'^\((.*)\)$|^\[(.*)\]$', r'\1\2', var)
        if var not in candidates and not var.isdigit() and not var.isalpha() and len(var.split()) <= 3:
            extend_vars.append(var)
    return extend_vars