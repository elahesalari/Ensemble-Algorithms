# 🎲 Ensemble Algorithms: Bagging & AdaBoost.M1

This project implements two popular ensemble learning algorithms — **Bagging** and **AdaBoost.M1** — using decision trees as base classifiers. The goal is to evaluate their robustness to Gaussian noise and tune their hyperparameters for optimal performance.

---

## 🧰 Methodology

- The base learner for both ensemble methods is the **DecisionTreeClassifier** from scikit-learn (`sklearn.tree.DecisionTreeClassifier`).  
- For **AdaBoost.M1**, the `max_depth` of the decision trees was experimentally tuned so that the base learner performs slightly better than random guessing.  
- For **Bagging**, default parameters of the decision tree were used without tuning.  

---

## ⚙️ Experimental Setup

- Gaussian noise \(\mathcal{N}(0,1)\) was added randomly to **10%**, **20%**, and **30%** of the features in each dataset.  
- Each dataset was split into **70% training** and **30% testing** sets.  
- The algorithms were run for **10 independent runs**, and average test accuracies were recorded for each noise level.  

---

## 🔢 Hyperparameter Tuning

- The number of iterations \(T\) was tested over the following sets:
  - Bagging: \(\{11, 21, 31, 41\}\)  
  - AdaBoost.M1: \(\{21, 31, 41, 51\}\)  
- Performance was evaluated for each \(T\), and the best results were reported.

---

## 📊 Results

### Ensemble performance under different noise levels and iteration numbers:

<div align="center">

![Bagging Accuracy](https://user-images.githubusercontent.com/91370511/159650282-4e004910-2311-4324-bdf2-9cabe47bf3f9.png)  
*Bagging: Test Accuracy vs Noise and Iterations*

![AdaBoost Accuracy](https://user-images.githubusercontent.com/91370511/159650410-08641efe-5420-4c54-b2ec-694673f5421c.png)  
*AdaBoost.M1: Test Accuracy vs Noise and Iterations*

![Comparison](https://user-images.githubusercontent.com/91370511/159650534-af96eca0-eccb-435c-806c-5670dff3e440.png)  
*Performance Comparison of Bagging and AdaBoost*

</div>




