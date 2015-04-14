class Animal:

    def __init__(self,legs,eyes):
        self.legs = legs
        self.eyes = eyes
        self.author = 'Joe'

    def number_of_legs(self):
        return self.legs

spider = Animal(8,8)

print spider.number_of_legs()

print spider.legs