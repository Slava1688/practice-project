from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from useful_package import polynom_3
from useful_package import hyperbola
import numpy

X = numpy.array([2, 7, 3, 5, 1, 4, 9])
y_poly = polynom_3(X)
y_hyp = hyperbola(X)

X = X.reshape(-1, 1)
model_poly = RandomForestRegressor()
model_poly.fit(X, y_poly)

model_hyp = RandomForestRegressor()
model_hyp.fit(X, y_hyp)

predict_poly = model_poly.predict(X)
predict_hyp = model_hyp.predict(X)

print(mean_squared_error(y_hyp, predict_hyp))
print(mean_squared_error(y_poly, predict_poly))

