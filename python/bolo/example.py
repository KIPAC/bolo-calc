"""Example class, feel free to remove"""

class Example:
    '''
    Example class for bolo package.
    '''
    def __init__(self, message):
        """Docstring for constructor"""
        self.message = message

    def run(self, raise_error=False):
        """Docstring for method"""
        if raise_error:
            raise RuntimeError()
        return self.message

    def get_message(self):
        """Docstring for method"""
        return self.message
