def get_text(name):
   return "lorem ipsum, {0} dolor sit amet".format(name)


# this is a decorator
def p_decorate(func):                               # The decorator takes a function (with its own argument) as argument...
   def func_wrapper(name):                          # ...and generates a new function...
       return "<p>{0}</p>".format(func(name))       # ...that takes the original function and augments its output
   return func_wrapper                              # ...returns the generated function so it can be used anywahere

# Create an instance of the wrapper function by calling the decorator; it has the get_text function as its argument
my_get_text = p_decorate(get_text)
print (my_get_text)


# Then supply the name as argument to this instance
print (my_get_text("John"))

# Decorate get_text by assigning the output of p_decorate - called on get_text - to get_text
get_text = p_decorate(get_text)
print(get_text('John'))


# Decorator syntax is a shortcut to avoid assigning decorator function on get_text to get_text
# Mention decorator function with @, then function to be decorated

@p_decorate
def get_text(name):
    return "lorem ipsum, {0} dolor sit amet".format(name)

print (get_text("John"))

# Other decorator functions may be wanted to wrap different tags around an output

def strong_decorate(func):
    def func_wrapper(name):
        return "<strong>{}</strong>".format(func(name))
    return func_wrapper

def div_decorate(func):
    def func_wrapper(name):
        return "<div>{0}</div>".format(func(name))
    return func_wrapper

# this avoids nesting function calls
@strong_decorate
@div_decorate
@p_decorate
def get_text(name):
    return "lorem ipsum, {0} dolor sit amet".format(name)
print (get_text("John"))



# Class definitions contain methods - functions that expect first param to be a reference to current object
# These can be decorated too

def p_decorate2(func):
   def func_wrapper(self):
       return "<p>{0}</p>".format(func(self))
   return func_wrapper


class Person(object):
    def __init__(self):
        self.name="John"
        self.family = "Doe"


    @p_decorate2
    def get_fullname(self):
        return self.name +" "+ self.family

my_person = Person()
print(my_person.get_fullname())

def p_decorate3(func):
   def func_wrapper(*args, **kwargs):
       return "<p>{0}</p>".format(func(*args,**kwargs))
   return func_wrapper

class Person(object):
    def __init__(self):
        self.name = "John"
        self.family = "Doe"

    @p_decorate
    def get_fullname(self):
        return self.name+" "+self.family

my_person=Person()
print (my_person.get_fullname())