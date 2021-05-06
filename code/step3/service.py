from model.model import make_model

class Service :
    #삼성전자, sk하이닉스, 현대차, LG화학, 카카오, 네이버
    STOCK_CODES = { 'samsung':'005930', 'sk':'000660', 'hyundai':'005380', 'lg':'051910', 'kakao':'035720',  'naver':'035420'}

    def __init__(self):
        pass

    def add_corporation(self, name, code):
        Service.STOCK_CODES[name.lower()] = str(code)

    def makeModel(self, name):
        self.__code = Service.STOCK_CODES[name.lower()]
        make_model(self.__code)


# add_corporation test
# if __name__ == '__main__' :
#     tmp = Service()
#     tmp.add_corporation('TEst','1111')
#     print(tmp.STOCK_CODES)

# makeModel test
if __name__ == '__main__' :
    tmp = Service()
    for name in tmp.STOCK_CODES :
        print('Start', name)
        tmp.makeModel(name)
        print('Finish', name)
        exit(0)