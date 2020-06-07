class method_info:
    def __init__(self):
        self.version = 0
        self.method = None

    def set_mothod(self, method_SAVE):
        self.method = method_SAVE
        self.version += 1