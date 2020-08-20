def decore(func):
    def vvv(value):
        print(1+value)
        func(value)
        print(2+value)
    return vvv

@decore
def medium(value):
    print(1.5)

medium(3)