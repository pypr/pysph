"""Utilities to work with the Python AST.
"""

import ast


class NameLister(ast.NodeVisitor):
    """Utility class to collect the Names in an AST.
    """
    def __init__(self, ctx=(ast.Load, ast.Store)):
        self.names = set()
        self.ctx = ctx

    def visit_Name(self, node):
        if isinstance(node.ctx, self.ctx):
            self.names.add(node.id)
        self.generic_visit(node)


class AugAssignLister(ast.NodeVisitor):
    """Utility class to collect the augmented assignments.
    """
    def __init__(self):
        self.names = set()

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.names.add(node.target.id)
        elif isinstance(node.target, ast.Subscript):
            self.names.add(node.target.value.id)
        self.generic_visit(node)

class AssignLister(ast.NodeVisitor):
    """Utility class to collect the augmented assignments.
    """
    def __init__(self):
        self.names = set()

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.names.add(target.id)
            elif isinstance(target, ast.Subscript):
                n = target.value
                while not isinstance(n, ast.Name):
                    n = n.value
                self.names.add(n.id)
            elif isinstance(target, (ast.List, ast.Tuple)):
                for n in target.elts:
                    if isinstance(n, ast.Name):
                        self.names.add(n.id)
        self.generic_visit(node)


def get_symbols(code, ctx=(ast.Load, ast.Store)):
    """Given an AST or code string return the symbols used therein.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    ctx: The context of the names, can be one of ast.Load, ast.Store, ast.Del.
    """
    if isinstance(code, str):
        tree = ast.parse(code)
    else:
        tree = code
    n = NameLister(ctx=ctx)
    n.visit(tree)
    return n.names

def get_aug_assign_symbols(code):

    """Given an AST or code string return the symbols that are augmented
    assign.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    """
    if isinstance(code, str):
        tree = ast.parse(code)
    else:
        tree = code
    n = AugAssignLister()
    n.visit(tree)
    return n.names

def get_assigned(code):
    """Given an AST or code string return the symbols that are augmented
    assigned or assigned.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    """
    if isinstance(code, str):
        tree = ast.parse(code)
    else:
        tree = code
    result = set()
    for l in (AugAssignLister(), AssignLister()):
        l.visit(tree)
        result.update(l.names)

    return result

def has_node(code, node):
    """Given an AST or code string returns True if the code contains
    any particular node statement.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    node: A node type or tuple of node types to check for.  If a tuple is passed
    it returns True if any one of them is in the code.
    """
    tree = ast.parse(code) if isinstance(code, str) else code
    for n in ast.walk(tree):
        if isinstance(n, node):
            return True
    return False

def has_return(code):
    """Returns True of the node has a return statement.
    """
    return has_node(code, ast.Return)
