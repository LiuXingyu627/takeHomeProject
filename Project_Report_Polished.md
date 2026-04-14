# Optimizing Marketing Spend through Income Prediction and Segmentation
**By [Xingyu Liu]**
---

## 1. Summary

This project delivers a dual-engine machine learning solution designed to optimize a retail client's targeted marketing strategy. Moving beyond a standard predictive score, this framework addresses two fundamental business questions:

1. **Maximizing Marketing ROI:** We built a predictive ranking engine to pinpoint customers most likely to earn over $50K. This allows the business to stop spending money on unlikely buyers and focus the budget where the purchasing power is.
2. **Unlocking Buying Behavior:** We grouped the population into distinct behavioral profiles. By understanding what drives these groups, we can match specific financial products to their actual lifestyle needs, drastically improving our conversion rates.

**Key Business Outcome:** The classification engine (XGBoost) achieved a highly robust weighted ROC-AUC of ~0.95. When paired with the K-Means segmentation strategy, the client can transition from generic blanket campaigns to a precision-targeted approach, maximizing ROI while maintaining clear governance through measurable KPIs.

---

## 2. Data Exploration and Preprocessing 

Our analysis utilized a census dataset of `~199,523` records containing 40 demographic and employment features. 

### Exploratory Findings & Business Judgments
* **Finding the Target Audience:** Only about 6% of the population in this dataset earns over $50K. Because high earners are so rare, standard "accuracy" is a misleading metric. If a model simply guessed "under $50K" for every single person, it would technically be 94% accurate, but entirely useless for finding high-value customers. We optimized for ranking metrics instead.
* **Real-World Weighting:** The data included a `weight` variable indicating how many actual US citizens a single record represents. We made the crucial business decision to apply these weights to all our training and testing. We are modeling the true US economy, not just a survey sample.

### Pre-Processing Pipeline
* **Simplifying the Data:** We stripped out overly complex variables to keep the model understandable for the business team. We want marketing teams to easily understand *why* the model makes a choice, using easy words rather than complex anomalies.
* **Formatting:** We converted the income labels into a simple binary format (1 for >$50K, 0 for under). To handle text categories like job titles or education levels, we used a process that translates words into simple "Yes/No" data points. For example, since an algorithm cannot read the word "Manager," we created a separate, dedicated column for every specific job title. If a customer is a Manager, they receive a 1 in the Manager column and a 0 in all others. This perfectly translates behavioral and demographic text into clean numbers that the algorithms can easily weigh and process.
* **Strict Validation:** We split our data into three locked vaults (Train, Validate, and Test). The models were trained on the first vault, tuned on the second, and took their "final exam" on the third to prove they work on completely unseen data.

---

## 3. Objective 1: Classification Engine for Target Prioritization

Our first objective is to locate high-value customers. To solve this, we built a predictive classification model that serves as our ranking engine. Instead of making a simple yes/no guess, the engine assigns every customer a probability score, allowing the business to rank them from most to least likely to buy.

### Methodology: How the Ranking Engine Works
We evaluated a progression of models and ultimately selected **XGBoost**. 

* **The Methodology Explanation:** Rather than building one massive, rigid set of rules, XGBoost builds a sequence of hundreds of simple decision-making frameworks. Each new framework evaluates the mistakes made by the previous ones and specifically focuses on correcting them. This "learning from past mistakes" approach makes it incredibly powerful at identifying subtle buying behaviors hidden deep within demographic data. 
* **Training Strategy:** We rigorously fine-tuned the engine's internal settings in an isolated testing environment to squeeze out maximum performance, then retrained it on all available learning data before its final evaluation.

### Visualizing Business Value and Strategic Decision-Making
When tested on our final, unseen hold-out data, the XGBoost engine maintained exceptional performance, proving its reliability for live campaigns.

![Classification Model Comparison](classification_auc_comparison.png)
> **Figure 1: Model Selection and Ranking Power.** > * **How to read this chart:** The bars represent the ranking power (ROC-AUC score) of four different algorithms. A score closer to 1.0 means the model is perfectly separating high-earners from low-earners.
> * **Business Decision Insight:** XGBoost outperformed the simpler baselines, achieving an impressive ~0.95 score. The business can confidently use this engine knowing it reliably identifies the right targets without simply memorizing historical data.

Marketing campaigns have different costs. A physical direct-mail campaign is expensive, while an email blast is nearly free. We designed the model to be flexible so the marketing team can adjust their targeting net based on their exact budget:

![Precision/Recall Tradeoff by Threshold](classification_threshold_tradeoff.png)
> **Figure 2: The Strategy Lever (Precision vs. Recall).** > * **How to read this chart:** The X-axis is the "confidence threshold" we set for the model. The blue line (Precision) shows how often we are right when we spend money to target someone. The orange line (Recall) shows what percentage of the total high-earning market we are capturing.
> * **Business Decision Insight:** This chart is the ultimate budget dial. If we have a limited budget for an expensive campaign, we move the dial to the right (high threshold) to ensure we only spend money on the most highly qualified leads. If we want to drive massive growth with a cheap digital campaign, we move the dial to the left (low threshold) to cast a wider net.

![Targeting Lift by Threshold](classification_lift_by_threshold.png)
> **Figure 3: Campaign Efficiency (Lift).** > * **How to read this chart:** "Lift" measures how much better our model is compared to randomly guessing. The X-axis represents the top percentage of our customer list we choose to target.
> * **Business Decision Insight:** If we use this model to target the top 10% of our scored customer list, we are roughly 14 times more likely to reach a high-earning prospect than if we picked names randomly. This visual proves the direct reduction in Customer Acquisition Cost (CAC).

---

## 4. Objective 2: Segmentation Engine to Offer Personalization

Knowing *who* is likely to have high income does not tell us *what* strategy they should receive. To drive business development, we must deeply understand customer buying behavior. We deployed a grouping engine to find natural, shared lifestyle patterns within the population.

### Methodology: How the Grouping Engine Works
* **Model Architecture & Selection:** We tested several heavy mathematical algorithms, including Agglomerative Hierarchical clustering and Gaussian Mixture Models (GMM). We ultimately selected **K-Means clustering** as our primary architecture because it provided the most actionable business groupings. 
* **Evaluation Procedure:** We did not guess how many groups to make. We utilized rigorous mathematical checks—including the visual "Elbow Method," corroborated by advanced separation metrics—to prove that exactly 5 personas is the optimal operational strategy. Any more would make campaign management too complex; any less would make the messaging too broad.
* **Opening the Black Box:** Clustering algorithms do not natively explain their choices. To understand the buying behavior driving these 5 groups, we built a secondary translator model (a Random Forest surrogate). This allowed us to extract the exact feature importances and see which demographic traits dictated the boundaries of each segment.

### Visualizing Business Value and Strategic Decision-Making

![Model Robustness](segmentation_model_robustness.png)
> **Figure 4: Infrastructure Quality Check (Clustering Robustness).** > * **How to read this chart:** This compares our chosen K-Means engine against other complex algorithms (Hierarchical and GMM). We look for the highest cluster quality and separation (blue bars), balanced against computational speed (red line).
> * **Business Decision Insight:** K-Means delivered the absolute best balance of crisp, high-quality groupings with the lowest computational cost. This ensures our IT and marketing infrastructure will not be bogged down by slow processing times while still delivering highly accurate personas.

![Top 10 Features Driving the Customer Segments](segmentation_embedded_2.png)
> **Figure 5: Persona Drivers.** > * **How to read this chart:** The longer the bar, the more influence that specific trait had on deciding which group a customer was placed into.
> * **Business Decision Insight:** Age, sex, and career status were the absolute strongest indicators separating the different buying behaviors of our customers. We can use this to brief our creative agencies on exactly who to feature in the marketing imagery.

We then profiled these clusters using weighted population data to map out our distinct marketing opportunities.

![Segment Opportunity Matrix](segmentation_opportunity_matrix.png)
> **Figure 6: Segment Opportunity Matrix.** > * **How to read this chart:** The X-axis is the size of the customer group (Population Share). The Y-axis is their earning power (Income >50K Rate). The size of the bubble visually represents the total population volume.
> * **Business Decision Insight:** This matrix visualizes the strategic priority of each segment and acts as our operational roadmap. Marketing budgets should be heavily weighted toward the top-right quadrant (High Population Share + High >$50K Rate).

### The Behavioral Personas & Product Strategy
By analyzing the matrix and the underlying demographic drivers, we identified distinct marketing opportunities to drive conversion across all 5 segments:

* **Segment 1 (The Primary Target):** This group represents 26.27% of the overall population but holds a massive 17.30% concentration of high earners. Targeting this group generates a 2.70x efficiency lift over the baseline population. 
  * **Buying Behavior Strategy:** This is our highest opportunity index. Direct the majority of premium marketing spend here with high-end financial products, luxury travel credit cards, and wealth management messaging.

* **Segments 2 & 4 (The Administrative Core):** These two segments are highly similar, together representing nearly 50% of the total population. They skew toward their late-30s, are primarily high school educated, and predominantly work in administrative support and clerical roles. They hold a moderate concentration of high earners (roughly 4% to 6%).
  * **Buying Behavior Strategy:** This is our core middle-class consumer base. Focus marketing efforts on everyday financial stability. Target them with no-fee checking accounts, standard cash-back credit cards (for groceries and gas), and competitive personal loans for debt consolidation.

* **Segment 3 (The Retirees/Transitionals):** This group skews older with lower active-work indicators and largely falls outside the traditional active workforce. 
  * **Buying Behavior Strategy:** Pivot messaging entirely away from aggressive credit building. Focus on fixed-income, high-yield CDs, and value-based asset preservation products.

* **Segment 0 (Low Priority / Dependents):** A massive population cluster but features near-zero direct high-income potential (this group naturally captured children and non-working dependents). 
  * **Buying Behavior Strategy:** Save marketing dollars by completely excluding this segment from premium direct outreach, or pivot to market 529 College Savings Plans directly to their parents.

---

## 5. Integrated Operating Model

We recommend moving away from a single, global campaign in favor of a two-layered deployment:

1. **Strategy Layer (Segmentation):** Assign each customer to a segment to determine the creative asset and product mix (e.g., Luxury vs. Value).
2. **Execution Layer (Classification):** Rank customers within their assigned segment. Apply a specific score threshold based on the budget allocated to that segment to maximize recall without sacrificing precision.

---

## 6. Dashboard & Governance Proposal

To transition this model from an analytical project to a living business tool, we recommend implementing a monthly KPI dashboard:

* **Segment KPIs:** Monitor month-over-month shifts in segment size and income concentration.
* **Drift Monitoring:** Establish a baseline for the classification score distribution and monitor for feature drift, ensuring the model's predictive power does not silently decay.
* **Fairness Audits:** Regularly analyze false-negative concentrations across protected demographic subgroups to ensure equitable marketing practices.

---

## 7. Limitations and Next Steps

* **Historical Snapshot:** The current models are trained on historical census data. They must be recalibrated against current, live data prior to full production deployment.
* **Proxy Variables:** Certain demographic features may act as proxies for socioeconomic status. Ongoing fairness monitoring is legally and ethically required.
* **Pilot Recommendation:** We recommend an immediate A/B test (Pilot) applying the Model-Targeted Strategy against a Control Group to measure realized revenue lift before scaling globally.

---

## 8. References
* Scikit-learn documentation: https://scikit-learn.org/stable/
* XGBoost documentation: https://xgboost.readthedocs.io/
* NumPy documentation: https://numpy.org/doc/
* pandas documentation: https://pandas.pydata.org/docs/