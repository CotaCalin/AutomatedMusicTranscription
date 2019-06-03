
class LoggerFactory:

    @staticmethod
    def getLogger(packageName):
        return Logger(packageName)

class Logger:
    def __init__(self, packageName):
        self.__packageName = packageName

    def logInfo(self, message):
        self.log(message, "INFO")

    def logError(self, message):
        self.log(message, "ERROR")

    def log(self, message, type):
        print("[%s] %s - %s" % (type, self.__packageName, message))
