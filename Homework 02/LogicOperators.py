class AND:
    def __init__(self):
        self.name = "AND"

    def result(self, x1, x2):
        return x1 and x2

class OR:
    def __init__(self):
        self.name = "OR"

    def result(self, x1, x2):
        return x1 or x2

class NAND:
    def __init__(self):
        self.name = "NAND"

    def result(self, x1, x2):
        return not (x1 and x2)

class NOR:
    def __init__(self):
        self.name = "NOR"

    def result(self, x1, x2):
        return not (x1 or x2)

class NOT:
    def __init__(self):
        self.name = "NOT"

    def result(self, x1, x2):
        return not x1

class XOR:
    def __init__(self):
        self.name = "XOR"

    def result(self, x1, x2):
        return (x1 and not x2) or (not x1 and x2)

class EQUAL:
    def __init__(self):
        self.name = "EQUAL"

    def result(self, x1, x2):
        return x1 == x2
