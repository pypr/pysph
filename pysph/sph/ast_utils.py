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


def get_symbols(code, ctx=(ast.Load, ast.Store)):
    """Given an AST or code string return the symbols used therein.
    
    Parameters
    ----------
    
    code: A code string or the result of an ast.parse.
    
    ctx: The context of the names, can be one of ast.Load, ast.Store, ast.Del.
    """
    if isinstance(code, basestring):
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
    if isinstance(code, basestring):
        tree = ast.parse(code)
    else:
        tree = code
    n = AugAssignLister()
    n.visit(tree)
    return n.names
