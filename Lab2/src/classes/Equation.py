import operator
import re
import math


class EquationTree:
    OPERATORS = {
        '+': (operator.add, 1),
        '-': (operator.sub, 1),
        '*': (operator.mul, 2),
        '/': (operator.truediv, 2),
        '^': (operator.pow, 3),
    }

    FUNCTIONS = {
        'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt'
    }

    class Node:
        def __init__(self, value, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right

        def __repr__(self):
            return f"Node({self.value})"

        def copy(self):
            left_copy = self.left.copy() if self.left else None
            right_copy = self.right.copy() if self.right else None
            return EquationTree.Node(self.value, left_copy, right_copy)

    def __init__(self, equation_str):
        self.original_equation = equation_str
        self.root = None
        self.variables = set()
        self._build_tree(equation_str)

    @staticmethod
    def _tokenize(equation_str):
        equation_str = equation_str.replace(' ', '')
        pattern = r'(?P<number>\d+\.?\d*)|(?P<variable>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<operator>[+\-*/^()])'
        tokens = []
        for match in re.finditer(pattern, equation_str, re.VERBOSE):
            token = match.group()
            tokens.append(token)
        return tokens

    @staticmethod
    def _is_number(token):
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _shunting_yard(self, tokens):
        output = []
        stack = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if self._is_number(token):
                output.append(float(token))
            elif token in self.variables or (token.isalpha() and token not in self.FUNCTIONS):
                output.append(token)
                self.variables.add(token)
            elif token in self.FUNCTIONS:
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack and stack[-1] == '(':
                    stack.pop()
                if stack and stack[-1] in self.FUNCTIONS:
                    output.append(stack.pop())
            elif token in self.OPERATORS:
                if token == '-' and (i == 0 or tokens[i - 1] == '(' or tokens[i - 1] in self.OPERATORS):
                    stack.append('unary-')
                else:
                    while (stack and stack[-1] != '(' and
                           stack[-1] in self.OPERATORS and
                           self.OPERATORS[stack[-1]][1] >= self.OPERATORS[token][1]):
                        output.append(stack.pop())
                    stack.append(token)
            i += 1
        while stack:
            output.append(stack.pop())
        return output

    def _build_tree_from_rpn(self, rpn):
        stack = []
        for token in rpn:
            if token == 'unary-':
                operand = stack.pop()
                node = self.Node('unary-', right=operand)
                stack.append(node)
            elif token in self.OPERATORS:
                right = stack.pop()
                left = stack.pop()
                node = self.Node(token, left, right)
                stack.append(node)
            elif token in self.FUNCTIONS:
                operand = stack.pop()
                node = self.Node(token, right=operand)
                stack.append(node)
            else:
                stack.append(self.Node(token))
        return stack[0] if stack else None

    def _build_tree(self, equation_str):
        tokens = self._tokenize(equation_str)
        rpn = self._shunting_yard(tokens)
        self.root = self._build_tree_from_rpn(rpn)

    def evaluate(self, **variables):
        def _eval_node(node):
            if node is None:
                return 0
            if isinstance(node.value, (int, float)):
                return node.value
            if node.value in variables:
                return variables[node.value]
            if node.value == 'unary-':
                return -_eval_node(node.right)
            if node.value in self.OPERATORS:
                left_val = _eval_node(node.left)
                right_val = _eval_node(node.right)
                return self.OPERATORS[node.value][0](left_val, right_val)
            if node.value in self.FUNCTIONS:
                arg = _eval_node(node.right)
                if node.value == 'sin':
                    return math.sin(arg)
                if node.value == 'cos':
                    return math.cos(arg)
                if node.value == 'tan':
                    return math.tan(arg)
                if node.value == 'log':
                    return math.log10(arg)
                if node.value == 'ln':
                    return math.log(arg)
                if node.value == 'exp':
                    return math.exp(arg)
                if node.value == 'sqrt':
                    return math.sqrt(arg)
            raise ValueError(f"Неизвестный узел: {node.value}")

        return _eval_node(self.root)

    def to_string(self):
        def _node_to_string(node):
            if node is None:
                return ""
            if isinstance(node.value, (int, float)):
                return str(node.value)
            if (node.value in self.variables or
                    (isinstance(node.value, str) and
                     node.value not in self.OPERATORS and
                     node.value not in self.FUNCTIONS and
                     node.value != 'unary-')):
                return node.value
            if node.value == 'unary-':
                return f"-({_node_to_string(node.right)})"
            if node.value in self.FUNCTIONS:
                return f"{node.value}({_node_to_string(node.right)})"
            if node.value in self.OPERATORS:
                left_str = _node_to_string(node.left)
                right_str = _node_to_string(node.right)
                return f"({left_str} {node.value} {right_str})"
            return str(node.value)

        result = _node_to_string(self.root)
        return result[1:-1] if result.startswith('(') and result.endswith(')') else result

    def get_variables(self):
        return self.variables

    def __repr__(self):
        return f"EquationTree('{self.original_equation}')"

    def print_tree(self, node=None, level=0, prefix=""):
        if node is None:
            node = self.root
        if level == 0:
            print(f"Уравнение: {self.original_equation}")
            print("Дерево:")

        if node.left is not None:
            self.print_tree(node.left, level + 1, "/ ")
        print(" " * (level * 4) + prefix + str(node.value))
        if node.right is not None:
            self.print_tree(node.right, level + 1, "\\ ")

    def get_terms(self):
        terms = []

        def _collect_terms(node, is_positive=True):
            if node is None:
                return
            if node.value in ['+', '-']:
                _collect_terms(node.left, is_positive)
                right_positive = is_positive if node.value == '+' else not is_positive
                _collect_terms(node.right, right_positive)
            else:
                if is_positive:
                    terms.append(node)
                else:
                    terms.append(self.Node('unary-', right=node))

        _collect_terms(self.root, True)
        return terms

    def terms_iterator(self):
        return EquationTermsIterator(self)

    def __iter__(self):
        return EquationTermsIterator(self)

    def derivative(self, variable):

        def _derivative_node(node, var):

            if node is None:
                return self.Node(0)

            if isinstance(node.value, (int, float)):
                return self.Node(0)

            if (isinstance(node.value, str) and
                    node.value not in self.OPERATORS and
                    node.value not in self.FUNCTIONS and
                    node.value != 'unary-'):

                if node.value == var:
                    return self.Node(1)
                else:
                    return self.Node(0)

            if node.value == 'unary-':
                return self.Node('unary-', right=_derivative_node(node.right, var))

            if node.value == '+':
                return self.Node('+',
                                 _derivative_node(node.left, var),
                                 _derivative_node(node.right, var))

            if node.value == '-':
                return self.Node('-',
                                 _derivative_node(node.left, var),
                                 _derivative_node(node.right, var))

            if node.value == '*':
                left_deriv = _derivative_node(node.left, var)
                right_deriv = _derivative_node(node.right, var)

                term1 = self.Node('*', left_deriv, node.right.copy())
                term2 = self.Node('*', node.left.copy(), right_deriv)

                return self.Node('+', term1, term2)

            if node.value == '/':
                u = node.left.copy()
                v = node.right.copy()
                u_prime = _derivative_node(node.left, var)
                v_prime = _derivative_node(node.right, var)

                numerator_left = self.Node('*', u_prime, v.copy())
                numerator_right = self.Node('*', u.copy(), v_prime)
                numerator = self.Node('-', numerator_left, numerator_right)
                denominator = self.Node('^', v.copy(), self.Node(2))

                return self.Node('/', numerator, denominator)

            if node.value == '^':
                base = node.left.copy()
                exponent = node.right.copy()

                if (isinstance(exponent.value, (int, float)) and
                        base.value == var):
                    new_exponent = self.Node(exponent.value - 1)
                    coeff = self.Node(exponent.value)
                    power = self.Node('^', base.copy(), new_exponent)
                    return self.Node('*', coeff, power)

                if (isinstance(base.value, (int, float)) and
                        exponent.value == var):
                    return self.Node('*',
                                     self.Node('ln', right=base.copy()),
                                     self.Node('^', base.copy(), exponent.copy()))

                u_prime = _derivative_node(node.left, var)
                v_prime = _derivative_node(node.right, var)

                term1 = self.Node('*',
                                  exponent.copy(),
                                  self.Node('^', base.copy(), self.Node('-', exponent.copy(), self.Node(1))))
                term1 = self.Node('*', term1, u_prime)

                term2 = self.Node('*',
                                  self.Node('^', base.copy(), exponent.copy()),
                                  self.Node('*',
                                            self.Node('ln', right=base.copy()),
                                            v_prime))

                return self.Node('+', term1, term2)

            if node.value in self.FUNCTIONS:
                arg = node.right.copy()
                arg_deriv = _derivative_node(node.right, var)

                match node.value:
                    case 'sin':
                        outer_deriv = self.Node('cos', right=arg.copy())
                    case 'cos':
                        outer_deriv = self.Node('unary-', right=self.Node('sin', right=arg.copy()))
                    case 'tan':
                        inner_cos = self.Node('cos', right=arg.copy())
                        outer_deriv = self.Node('^', inner_cos, self.Node(-2))
                    case 'log':
                        outer_deriv = self.Node('/', self.Node(1), self.Node('*', arg.copy(), self.Node(math.log(10))))
                    case 'ln':
                        outer_deriv = self.Node('/', self.Node(1), arg.copy())
                    case 'exp':
                        outer_deriv = self.Node('exp', right=arg.copy())
                    case 'sqrt':
                        outer_deriv = self.Node('/', self.Node(1),
                                                self.Node('*', self.Node(2), self.Node('sqrt', right=arg.copy())))
                    case _:
                        outer_deriv = 0

                return self.Node('*', outer_deriv, arg_deriv)

            return self.Node(0)

        derivative_root = _derivative_node(self.root, variable)
        derivative_eq = EquationTree("")
        derivative_eq.root = derivative_root
        derivative_eq.variables = self.variables.copy()
        derivative_eq.original_equation = derivative_eq.to_string()

        return derivative_eq

    def is_linear_term(self, term_node, variable=None):
        def _check_linearity(node, target_var):
            if node is None:
                return True, 0
            if isinstance(node.value, (int, float)):
                return True, 0
            if (isinstance(node.value, str) and
                    node.value not in self.OPERATORS and
                    node.value not in self.FUNCTIONS and
                    node.value != 'unary-'):
                if target_var is None:
                    return True, 1
                else:
                    return True, 1 if node.value == target_var else 0
            if node.value == 'unary-':
                return _check_linearity(node.right, target_var)
            if node.value == '*':
                left_linear, left_power = _check_linearity(node.left, target_var)
                right_linear, right_power = _check_linearity(node.right, target_var)
                if left_linear and right_linear:
                    total_power = left_power + right_power
                    return total_power <= 1, total_power
                return False, 0
            if node.value in ['+', '-']:
                return False, 0
            if node.value == '/':
                num_linear, num_power = _check_linearity(node.left, target_var)
                denom_linear, denom_power = _check_linearity(node.right, target_var)
                if num_linear and denom_linear and denom_power == 0:
                    return True, num_power
                return False, 0
            if node.value == '^':
                left_linear, left_power = _check_linearity(node.left, target_var)
                if (isinstance(node.right, self.Node) and
                        isinstance(node.right.value, (int, float)) and
                        node.right.value == 1):
                    return left_linear, left_power
                return False, 0
            if node.value in self.FUNCTIONS:
                return False, 0
            return False, 0

        is_linear, power = _check_linearity(term_node, variable)
        return is_linear and power <= 1

    def get_linear_terms(self, variable=None):
        linear_terms = []
        for term_node in self.get_terms():
            if self.is_linear_term(term_node, variable):
                linear_terms.append(term_node)
        return linear_terms

    def is_linear_equation(self, variable=None):
        for term_node in self.get_terms():
            if not self.is_linear_term(term_node, variable):
                return False
        return True

    def extract_linear_part(self, variable=None):
        linear_terms = self.get_linear_terms(variable)
        if not linear_terms:
            linear_eq = EquationTree("0")
            return linear_eq
        linear_root = self._build_subtraction_tree(self.Node(0), self._build_sum_tree(linear_terms))
        linear_eq = EquationTree("")
        linear_eq.root = linear_root
        linear_eq.variables = self.variables.copy()
        nonlinear_root = self._build_sum_tree([self.root, linear_root])
        nonlinear_eq = EquationTree("")
        nonlinear_eq.root = nonlinear_root
        nonlinear_eq.variables = self.variables.copy()
        return nonlinear_eq, linear_eq

    def _build_sum_tree(self, nodes):
        if not nodes:
            return self.Node(0)
        result = nodes[0]
        for node in nodes[1:]:
            result = self.Node('+', result, node)
        return result

    def _build_subtraction_tree(self, minuend, subtrahend):
        return self.Node('-', minuend, subtrahend)

    def copy(self):
        new_eq = EquationTree("")
        new_eq.root = self.root.copy() if self.root else None
        new_eq.variables = self.variables.copy()
        new_eq.original_equation = self.original_equation
        return new_eq


class EquationTermsIterator:
    def __init__(self, equation_tree):
        self.equation_tree = equation_tree
        self.terms = equation_tree.get_terms()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.terms):
            raise StopIteration
        term_node = self.terms[self.index]
        self.index += 1
        return self._node_to_term_string(term_node)

    def _node_to_term_string(self, node):
        def _node_to_string(node):
            if node is None:
                return ""
            if isinstance(node.value, (int, float)):
                return str(node.value)
            if (node.value in self.equation_tree.variables or
                    (isinstance(node.value, str) and
                     node.value not in self.equation_tree.OPERATORS and
                     node.value not in self.equation_tree.FUNCTIONS and
                     node.value != 'unary-')):
                return node.value
            if node.value == 'unary-':
                return f"-{_node_to_string(node.right)}"
            if node.value in self.equation_tree.FUNCTIONS:
                return f"{node.value}({_node_to_string(node.right)})"
            if node.value in self.equation_tree.OPERATORS:
                left_str = _node_to_string(node.left)
                right_str = _node_to_string(node.right)
                if node.value == '*':
                    if ('+' in left_str or '-' in left_str) and ('+' in right_str or '-' in right_str):
                        return f"({left_str})*({right_str})"
                    elif '+' in left_str or '-' in left_str:
                        return f"({left_str})*{right_str}"
                    elif '+' in right_str or '-' in right_str:
                        return f"{left_str}*({right_str})"
                    else:
                        return f"{left_str}*{right_str}"
                if node.value == '^':
                    return f"{left_str}^{right_str}"
                return f"({left_str} {node.value} {right_str})"
            return str(node.value)

        result = _node_to_string(node)
        if result.startswith('(') and result.endswith(')'):
            result = result[1:-1]
        return result
