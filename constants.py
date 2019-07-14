from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


MODEL_MAPPER = {
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor
    }