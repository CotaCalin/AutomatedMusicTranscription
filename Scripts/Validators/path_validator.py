import os

class PathValidator:
    def __init__(self, path, isDir=False):
        self.__path = path
        self.__isDir = isDir
        self.validate()

    def validate(self):
        isDir =  os.path.isdir(self.__path)
        exists = os.path.exists(self.__path)

        if not exists:
            raise Exception("{0} doesn't exist!".format(self.__path))

        if self.__isDir and not isDir:
            raise Exception("{0} is not a valid directory!".format(self.__path))
        elif isDir:
            raise Exception("{0} is a directory and it shouldn't be!".format(self.__path))
