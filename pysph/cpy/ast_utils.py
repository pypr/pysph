"""Utilities to work with the Python AST.
"""

import ast
import sys

PY_VER = sys.version_info.major


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


class SymbolParser(ast.NodeVisitor):
    """Utility class to gather the used symbols in a block of code. We look at
    assignments, augmented assignments, function calls, and any Names. These
    are all parsed in one shot and collected.

    Note that this works best for a single function that is parsed rather than
    for a collection of functions.

    """
    def __init__(self):
        self.names = set()
        self.assign = set()
        self.calls = set()
        self.funcargs = set()
        self.func_name = ''
        self.ctx = (ast.Load, ast.Store)

    def visit_Name(self, node):
        if isinstance(node.ctx, self.ctx):
            self.names.add(node.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.assign.add(node.target.id)
        elif isinstance(node.target, ast.Subscript):
            v = node.target.value
            while not isinstance(v, ast.Name):
                v = v.value
            self.assign.add(v.id)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assign.add(target.id)
            elif isinstance(target, ast.Subscript):
                n = target.value
                while not isinstance(n, ast.Name):
                    n = n.value
                self.assign.add(n.id)
            elif isinstance(target, (ast.List, ast.Tuple)):
                for n in target.elts:
                    if isinstance(n, ast.Name):
                        self.assign.add(n.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.func_name = node.name
        if PY_VER == 2:
            self.funcargs.update(x.id for x in node.args.args)
            if node.args.vararg:
                self.funcargs.add(node.args.vararg)
            if node.args.kwarg:
                self.funcargs.add(node.args.kwarg)
        else:
            self.funcargs.update(x.arg for x in node.args.args)
            if node.args.vararg:
                self.funcargs.add(node.args.vararg.arg)
            if node.args.kwarg:
                self.funcargs.add(node.args.kwarg.arg)
            if node.args.kwonlyargs:
                self.funcargs.update(x.arg for x in node.args.kwonlyargs)
        for arg in node.body:
            self.visit(arg)


def _get_tree(code):
    return ast.parse(code) if isinstance(code, str) else code


def get_symbols(code, ctx=(ast.Load, ast.Store)):
    """Given an AST or code string return the symbols used therein.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    ctx: The context of the names, can be one of ast.Load, ast.Store, ast.Del.
    """
    tree = _get_tree(code)
    n = NameLister(ctx=ctx)
    n.visit(tree)
    return n.names


def get_assigned(code):
    """Given an AST or code string return the symbols that are augmented
    assigned or assigned.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    """
    tree = _get_tree(code)
    p = SymbolParser()
    p.visit(tree)
    return p.assign


def get_unknown_names_and_calls(code):
    """Given an AST or code string return the unknown variables and calls in
    the code.  The function returns two sets, ``names, calls``.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    """
    tree = ast.parse(code) if isinstance(code, str) else code
    p = SymbolParser()
    p.visit(tree)
    funcargs = p.funcargs
    if len(p.func_name) > 0:
        funcargs.add(p.func_name)
    names = p.names - funcargs - p.calls - p.assign
    calls = p.calls
    return names, calls


def has_node(code, node):
    """Given an AST or code string returns True if the code contains
    any particular node statement.

    Parameters
    ----------

    code: A code string or the result of an ast.parse.

    node: A node type or tuple of node types to check for.  If a tuple
        is passed it returns True if any one of them is in the code.
    """
    tree = _get_tree(code)
    for n in ast.walk(tree):
        if isinstance(n, node):
            return True
    return False


def has_return(code):
    """Returns True of the node has a return statement.
    """
    return has_node(code, ast.Return)
