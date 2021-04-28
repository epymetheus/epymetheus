class Universe:
    def __init__(self, prices, name=None):
        self.prices = prices
        self.name = name
        raise DeprecationWarning("Universe is deprecated. Just use pandas.DataFrame.")
