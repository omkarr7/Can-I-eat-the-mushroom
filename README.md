# Mushroom Edibility Classification 🍄

## Aim
The aim of this project is to classify mushrooms as edible or poisonous based on various physical and olfactory characteristics using machine learning models. The dataset comes from The Audubon Society Field Guide to North American Mushrooms and includes features like cap shape, cap surface, cap color, odor, and height. The goal is to predict whether a mushroom is safe to eat.

## Dataset
- **Source**: The Audubon Society Field Guide to North American Mushrooms (1981)
- **Format**: CSV
- **Features**:
  - `Edible`: Edible or Poisonous
  - `CapShape`, `CapSurface`, `CapColor`: Physical characteristics
  - `Odor`: Mushroom smell (e.g., almond, foul)
  - `Height`: Tall or Short
## 🧪 Methodology

1. **Data Preprocessing**  
   Loaded the dataset, checked for missing values, and converted categorical variables into factors.

2. **Exploratory Data Analysis**  
   Visualized feature distributions and correlations. Identified ‘Odor’ as the most influential feature.

3. **Model Training**  
   Trained a Decision Tree and a Random Forest model on a 70/30 train-test split.

4. **Model Evaluation**  
   Assessed model accuracy and feature importance. Random Forest achieved the highest performance.

5. **Cross-Validation & Statistical Testing**  
   Performed 10-fold cross-validation and used a paired t-test to confirm that Random Forest significantly outperformed the Decision Tree.

## 📊 Results

- **Decision Tree Accuracy**: ~98.6%  
- **Random Forest Accuracy**: ~99.3%  
- **Most important feature**: `Odor`  
- **Best Model**: Random Forest (statistically significant improvement)

## Documentation
For complete analysis and conclusions, please refer to this **[Documentation](https://github.com/omkarr7/Can-I-eat-the-mushroom/edit/main/mushroom.R)**
