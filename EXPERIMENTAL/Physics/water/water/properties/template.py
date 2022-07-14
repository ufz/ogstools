import pathlib
class PROPERTY:
    def __init__(self):
        pass
    def value(self):
        raise NotImplementedError
    def dvalue(self):
        raise NotImplementedError
    def exprtk_value(self):
        raise NotImplementedError
    def exprtk_dvalue(self):
        raise NotImplementedError
    def _getcodeasstring(self, fct, variable):
        classname = self.__class__.__name__
        string = ""
        currentpath = pathlib.Path(__file__).parent.resolve()
        with open(f"{currentpath}/{classname.split('_')[0]}.py") as source:
            currentclass = None
            currentmethod = None
            currentvariable = None
            for line in source:
                linelist = line.split(" ")
                try:
                    if linelist[0] == "class":
                        currentclass = linelist[1].split("(")[0]
                        currentmethod = None
                        currentvariable = None
                except IndexError:
                    pass
                try:
                    if linelist[4] == "def":
                        currentmethod = linelist[5].split("(")[0]
                except IndexError:
                    pass
                try:
                    if (linelist[8] == "if") or (linelist[8] == "elif"):
                        if linelist[9] == "variable":
                            currentvariable = linelist[11].split('"')[1]
                except IndexError:
                    pass
                try:
                    if classname == currentclass:
                        if currentmethod == fct:
                            if currentvariable == variable:
                                if "return" in linelist:
                                    string = line.split("return")[1]
                except IndexError:
                    pass
        return string

    def _convertpythontoexprtk(self, string):
        string = string.replace("**", "^")
        string = string.replace("np.maximum","max")
        string = string.replace("np.minimum","min")
        string = string.replace("np.sign","sgn")
        string = string.replace("np.","")
        string = string.replace("\n","")
        return string
