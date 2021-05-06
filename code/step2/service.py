from model.model import make_model, prediction, make_data_for_tomorrow

class Service :
    #삼성전자, sk하이닉스, 현대차, LG화학, 카카오, 네이버
    STOCK_CODES = { 'samsung':'005930', 'sk':'000660', 'hyundai':'005380', 'lg':'051910', 'kakao':'035720',  'naver':'035420'}

    def __init__(self):
        pass

    def add_corporation(self, name, code):
        Service.STOCK_CODES[name.lower()] = str(code)

    # 모델만 만들어두고 싶을때
    def makeModel(self, name):
        self.__code = Service.STOCK_CODES[name.lower()]
        make_model(self.__code)

    # 실제 데이터 값을 넣고 예측 (만약 모델이 없다면 만들어준다.)
    def prediction(self, name, data):
        self.__code = Service.STOCK_CODES[name.lower()]
        self.__pred = prediction(self.__code, data)
        print(self.__pred)
        return self.__pred

    # 기업 이름을 넣으면 해당 기업의 내일 종가를 예측해준다.
    def predictTomorrow(self, name):
        self.__code = Service.STOCK_CODES[name.lower()]
        self.__data = make_data_for_tomorrow(self.__code)
        self.__pred = prediction(self.__code, self.__data )
        print(self.__pred)
        return self.__pred


# add_corporation test
# if __name__ == '__main__' :
#     tmp = Service()
#     tmp.add_corporation('TEst','1111')
#     print(tmp.STOCK_CODES)


# makeModel test
# if __name__ == '__main__' :
#     tmp = Service()
#     for name in tmp.STOCK_CODES :
#         print('Start', name)
#         tmp.makeModel(name)
#         print('Finish', name)


# # predict_tomorrow_test
# if __name__ == '__main__' :
#     tmp = Service()
#     for name in tmp.STOCK_CODES :
#         print('Start', name)
#         print(tmp.predictTomorrow(name))
#         print('Finish', name, '\n\n')

# makeModel test
if __name__ == '__main__' :
    tmp = Service()
    for name in tmp.STOCK_CODES :
        print('Start', name)
        tmp.makeModel(name)
        print('Finish', name)
        exit(0)


'''
    고민점
    1. class의 prediction 함수가 정말 필요한지 (특정 기간의 데이터를 넣으면 그 다음 데이터 출력 할거임, predict_tomorrow 함수와 같은데 다른 점은 내가 원하는 기간을 지정할 수 있다는 것)
    2. 종가 말고 다른 가격들( ex)시가, 고가 등 ) 을 이용해서 모델 만들고 설계 할수 있도록 cols를 input으로 받을건지
    3. 2번과 동일한 맥락으로 WINDOW_SIZE (며칠치를 이용해서 다음날 주가를 예측할건지) 도  input으로 받을건지
    4. 단순 LSTM 1개와 output layer 총 2개로 충분한지
    5. prediction 할때 inverse 할지 말지 결정해야함 (함수를 새로 만들까 flag 두어서 분기를 나눌까 생각 중)
'''
