import pickle
from sklearn import svm

pickle_in = open('allFactors.pickle', 'rb')

model = pickle.load(pickle_in)


def accident_level(day_of_week, time, long, lat, weather, light, road):
    return model.predict([[day_of_week, time, long, lat, weather, light, road]])

print(accident_level(5,0,149.138901726867,-35.3445942499744,7,3,3))

print('done!')
