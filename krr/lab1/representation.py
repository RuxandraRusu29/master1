import copy

class Expr:
    """
    Empty base class defining an expression in propositional logic.
    Propositions and operators that build expressions will extend from this class.
    """
    def negate(self) -> "Expr":
        raise NotImplementedError("Method not implemented")

    def eval(self) -> bool:
        raise NotImplementedError("Method not implemented")

    def simplify(self) -> "Expr":
        """
        The simplify method operates in two ways:
          1. it converts "->" and "<->" to expressions containing only AND, OR, NOT operators
          2. it eliminates the NOT operator from expressions, until it reaches propositions
        """
        return self

class Proposition(Expr):
    """
    Defines a term in propositional logic
    """

    def __init__(self, name: str, value: bool):
        self.name = name
        self.value = value

    def negate(self):
        return Proposition(self.name, not self.value)

    def eval(self):
        return self.value

    def is_negation_of(self, other: "Proposition"):
        return self == other.negate()

    def __hash__(self) -> int:
        name_hash = hash(self.name)
        if self.value:
            return name_hash + 1
        else:
            return name_hash

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __str__(self):
        if self.value:
            return self.name
        else:
            return "¬" + self.name

    def __repr__(self):
        return self.__str__()


class UnaryOperator(Expr):
    """
    Base class for a unary operator in propositional logic
    """
    def __init__(self, op_name: str, expression: Expr):
        self.op_name = op_name
        self.expression = expression

    def __str__(self):
        return "%s(%s)" % (self.op_name.upper(), str(self.expression))

    def __repr__(self):
        return self.__str__()


class Not(UnaryOperator):
    """
    Defines a negation operation in propositional logic
    """
    def __init__(self, expression):
        super(Not, self).__init__("NOT", expression)

    def negate(self):
        return self.expression

    def eval(self):
        return not self.expression.eval()

    def simplify(self):
        """
        Simplification for a NOT operator is pushing the negation inside the expression
        :return: The propositional logic expression resulting from pushing the negation operator inside the negated
        expression (i.e. removal of the negation operator)
        """
        return self.expression.negate()


class BinaryOperator(Expr):
    """
    Base class for a binary operator in propositional logic
    """
    def __init__(self, op_name: str, left_expr: Expr, right_expr: Expr):
        self.op_name = op_name
        self.left_expr = left_expr
        self.right_expr = right_expr

    def __str__(self):
        return "%s(%s, %s)" % (self.op_name.upper(), str(self.left_expr), str(self.right_expr))

    def __repr__(self):
        return self.__str__()


class And(BinaryOperator):
    """
    Defines an "AND" operator in propositional logic
    """
    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(And, self).__init__("AND", left_expr, right_expr)

    def eval(self):
        left_expr_eval = self.left_expr.eval()
        if not left_expr_eval:
            return False
        else:
            return self.right_expr.eval()

    def negate(self):
        return Or(self.left_expr.negate(), self.right_expr.negate())

    def simplify(self):
        return And(self.left_expr.simplify(), self.right_expr.simplify())


class Or(BinaryOperator):
    """
    Defines an "OR" operator in propositional logic
    """
    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(Or, self).__init__("OR", left_expr, right_expr)

    def eval(self):
        left_expr_eval = self.left_expr.eval()
        if left_expr_eval:
            return True
        else:
            return self.right_expr.eval()

    def negate(self):
        return And(self.left_expr.negate(), self.right_expr.negate())

    def simplify(self):
        return Or(self.left_expr.simplify(), self.right_expr.simplify())


class Implies(BinaryOperator):
    """
    Defines an "Implication" operator in propositional logic
    """
    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(Implies, self).__init__("IMPL", left_expr, right_expr)

    def eval(self):
        return self.simplify().eval()

    def negate(self):
        return And(self.left_expr, self.right_expr.negate())

    def simplify(self):
        return Or(self.left_expr.negate(), self.right_expr)


class Equivalent(BinaryOperator):
    """
    Defines an "Equivalence" operator in propositional logic
    """

    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(Equivalent, self).__init__("EQ", left_expr, right_expr)

    def eval(self):
        return self.simplify().eval()

    def negate(self):
        return Or(And(self.left_expr, self.right_expr.negate()), And(self.left_expr.negate(), self.right_expr))

    def simplify(self):
        return Or(And(self.left_expr, self.right_expr), And(self.left_expr.negate(), self.right_expr.negate()))

class SemanticTableauxNode():
    def __init__(self, expressions):
        self.expressions = expressions
        self.right_child = None
        self.left_child = None
        
    def check_valid(self):
        #print(self.expressions)
        for i in range(0, len(self.expressions) - 1):
            for j in range(i+1, len(self.expressions)):
                if type(self.expressions[i].simplify()) == Proposition and type(self.expressions[j].simplify()) == Proposition and self.expressions[i].simplify() == self.expressions[j].negate().simplify():
                    #print("found contradiction")
                    return False
        #print("is ok")
        return True
    
    def evaluate(self):
        print(self.expressions)
        if self.check_valid() == False:
            return False
        #print()
        for i in range(0, len(self.expressions)):
            #print(i)
            if isinstance(self.expressions[i], Proposition):
                continue
            if self.expressions[i].op_name == 'OR':
                self.left_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                self.right_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                self.left_child.expressions[i] = self.expressions[i].left_expr
                self.right_child.expressions[i] = self.expressions[i].right_expr
                check = self.left_child.evaluate()
                if check == True:
                    return True
                else:
                    return self.right_child.evaluate()
            elif self.expressions[i].op_name == 'AND':
                self.expressions.append(self.expressions[i].right_expr)
                self.expressions[i] = self.expressions[i].left_expr
                return self.evaluate()
            elif self.expressions[i].op_name == 'IMPL':
                #print("implies" + str(i))
                self.left_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                self.right_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                self.left_child.expressions[i] = Not(self.expressions[i].left_expr)
                self.right_child.expressions[i] = self.expressions[i].right_expr
                check = self.left_child.evaluate()
                if check == True:
                    return True
                else:
                    return self.right_child.evaluate()
            elif self.expressions[i].op_name == 'EQ':
                self.left_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                self.right_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                self.left_child.expressions[i] = And(self.expressions[i].left_expr, self.expressions[i].right_expr)
                self.right_child.expressions[i] = And(Not(self.expressions[i].left_expr), Not(self.expressions[i].right_expr))
                check = self.left_child.evaluate()
                if check == True:
                    return True
                else:
                    return self.right_child.evaluate()
            elif self.expressions[i].op_name == 'NOT':
                if isinstance(self.expressions[i].expression, Proposition):
                    continue
                if self.expressions[i].expression.op_name == 'NOT':
                    self.expressions[i] = self.expressions[i].expression.expression
                    return self.evaluate()
                elif self.expressions[i].expression.op_name == 'AND':
                    self.left_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                    self.right_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                    self.left_child.expressions[i] = Not(self.expressions[i].expresion.left_expr)
                    self.right_child.expressions[i] = Not(self.expressions[i].expresion.right_expr)
                    check = self.left_child.evaluate()
                    if check == True:
                        return True
                    else:
                        return self.right_child.evaluate()
                elif self.expressions[i].expression.op_name == 'EQ':
                    self.left_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                    self.right_child = SemanticTableauxNode(copy.deepcopy(self.expressions))
                    self.left_child.expressions[i] = And(self.expressions[i].left_expr, Not(self.expressions[i].right_expr))
                    self.right_child.expressions[i] = And(Not(self.expressions[i].left_expr), self.expressions[i].right_expr)
                    check = self.left_child.evaluate()
                    if check == True:
                        return True
                    else:
                        return self.right_child.evaluate()
                elif self.expressions[i].expression.op_name == 'OR':
                    self.expressions.append(Not(self.expressions[i].expression.right_expr))
                    self.expressions[i] = Not(self.expressions[i].expression.left_expr)
                    return self.evaluate()
                elif self.expressions[i].expression.op_name == 'IMPL':
                    #print("not impl")
                    self.expressions.append(Not(self.expressions[i].expression.right_expr))
                    self.expressions[i] = self.expressions[i].expression.left_expr
                    return self.evaluate()
        return True

if __name__ == "__main__":
    a1 = Proposition("A1", True)
    neg_a1 = Proposition("A1", False)

    a2 = Proposition("A2", True)
    a3 = Proposition("A3", True)

    assert a1.is_negation_of(neg_a1)
    assert neg_a1.is_negation_of(a1)

    assert a2 != a3
    assert a1 == neg_a1.negate()

    expr1 = Implies(a1, a2)
    expr2 = Or(a3, Not(Or(a3, neg_a1)))

    assert expr1.eval(), "a1->a2 must be true, since a1 and a2 are true"
    assert expr2.eval(), "a3 v ~(a3 v ~a1) must be true, since a3 is true"

    expr = Not(And(expr1, expr2))
    print("Truth value of %s is %r" % (str(expr), expr.eval()))

    print("Simplified version of expression \n\t%s is \n\t%s" % (str(expr), str(expr.simplify())))

    print("Negation of expression \n\t%s is \n\t%s" % (str(expr), str(expr.negate())))

    expr1 = Implies(Proposition("p", True), Proposition("q", True))
    expr2 = Implies(Proposition("q", True), Proposition("r", True))
    #expr3 = And(Proposition("p", True), Proposition("r", False))
    expr3 = Implies(Proposition("p", True), Proposition("r", True))

    semantic_tableaux = SemanticTableauxNode([expr1, expr2, expr3])
    print("something")
    print(semantic_tableaux.evaluate())
    
    expr1 = Implies(Proposition("s", True), Proposition("b", True))
    expr2 = Implies(Proposition("b", True), Proposition("d", True))
    expr3 = Implies(Proposition("d", True), Proposition("f", True))
    expr4 = And(Proposition("s", True), Not(Proposition("f", True)))

    semantic_tableaux = SemanticTableauxNode([expr1, expr2, expr3, expr4])
    print("Problem1")
    #print(semantic_tableaux.evaluate())

    expr1 = Implies(Proposition("pla", True), Proposition("pba", True))
    expr2 = Implies(Proposition("wla", True), Proposition("wba", True))
    expr3 = Implies(Proposition("sla", True), Proposition("sba", True))
    expr4 = Implies(Proposition("wba", True), Proposition("bha", True))
    expr5 = Or(Proposition("pla", True), Or(Proposition("wla", True), Proposition("sla", True)))
    expr6 = Proposition("bha", True)

    semantic_tableaux = SemanticTableauxNode([expr1, expr2, expr3, expr4, expr5, expr6])
    print("Problem2")
    print(semantic_tableaux.evaluate())
    
    expr1 = Implies(Proposition("yb", True), Proposition("yph", True))
    expr2 = Implies(Proposition("yph", True), Not(Proposition("yl", True)))
    expr3 = Implies(Proposition("ry", True), Proposition("yl", True))
    expr4 = And(Proposition("yb", True), Proposition("ry", True))

    #semantic_tableaux = SemanticTableauxNode([expr1, expr2, expr3, expr4])
    print("Problem3")
    #print(semantic_tableaux.evaluate())
    
    expr1 = Or(Proposition("ac", True), Proposition("ab", True))
    expr2 = Implies(Proposition("cgw", True), Not(Proposition("chg", True)))
    expr3 = Implies(Proposition("agw", True), And(Proposition("cgw", True), Proposition("ab", True)))
    expr4 = Implies(Proposition("agw", True), And(Proposition("ab", True), Proposition("bgw", True)))
    expr5 = Implies(Proposition("ab", True), Not(Proposition("cgw", True)))
    expr6 = Implies(Proposition("ac", True), Proposition("chg", True))
    expr7 = Implies(Proposition("bgw", True), Not(Proposition("cht", True)))
    expr8 = Not(Proposition("chg", True))
    expr9 = Proposition("cht", True)
    expr10 = Proposition("agw", True)
    
    #semantic_tableaux = SemanticTableauxNode([expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9, expr10])
    print("Problem4")
    #print(semantic_tableaux.evaluate())
