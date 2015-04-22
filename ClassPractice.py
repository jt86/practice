class Animal:
    "A class of object for livestock"
    kingdom = 'fauna'
    def __init__(self,legs,eyes):

        self.legs = legs
        self.eyes = eyes
        self.author = 'Joe'

    def number_of_legs_and_eyes(self):
        return self.legs+self.eyes

print Animal.__doc__

spider = Animal(8,8)
print spider.__doc__
print spider.number_of_legs_and_eyes()
print spider.legs
print spider.kingdom
print Animal.kingdom