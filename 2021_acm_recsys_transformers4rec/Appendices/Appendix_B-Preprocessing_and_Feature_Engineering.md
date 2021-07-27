# Appendix B - Datasets Preprocessing and Feature Engineering

First, the user interactions are grouped by sessions. All datasets provide session ids, except the ADRESSA dataset, for which we artificially split sessions to have a maximum idle time of 30 minutes between the user interactions.

Repeated user interactions on the same items within sessions are removed from news dataset, as they do not provide information gain. For the e-commerce domain, repeated interactions are common when users are comparing products and recommending items already interacted can be helpful from a user’s perspective e.g., as reminders [1,2,3,4]. Thus we remove consecutive interactions in the same items, but allow them to be interleaved, e.g. the sequence of interacted items 𝑎𝑎𝑎𝑏𝑐𝑐𝑎 becomes 𝑎𝑏𝑐𝑎. We remove sessions with length 1, and truncate sessions up to the maximum of 20 interactions.

The sessions are divided in time windows, according to the unit: one day for e-commerce datasets and one hour for the news datasets. The reason for is that interactions in the news domain are very biased toward recent items. For example, in G1 news, 50% and 90% of interactions are performed on articles published within up to 11 hours and 20hours, respectively. So, training those types of algorithms on a daily basis would not be effective for recommending fresh articles, as they would not be included in the train set.

We also explore the usage of side features by Transformers architectures (RQ3). The following table presents the additional features other than the item id and their feature engineering that were used to by our experiments to address RQ3, which explores different techniques to include side information into Transformers.
It is worthwhile to note that the YOOCHOOSE dataset have a single categorical feature (category), but it is inconsistent over time in the dataset. All interactions before day 84 (2014-06-23) have the same value for that feature, when many other categories are introduced. Under the incremental evaluation protocol, this drops significantly the model accuracy for the early subsequent days so we cannot use that feature for our purpose. As there was no other categorical feature left for the YOOCHOOSE dataset, we decided not including it for the analysis of RQ3.

<br>
<h4>Table 3. Datasets feature engineering</h4>
<font size="2" face="Arial" >
<table class="hp-table">
<thead><tr class="table-firstrow"><th></th><th>REES46 eCommerce</th><th>G1 news</th><th>Adressa news</th><th>Preprocessing techniques</th></tr></thead><tbody>
 <tr><td>Categorical features</td><td>category, subcategory, brand</td><td>User context features: region, country, environment, device group, OS</td><td>category, subcategory, author and user context features: city, region, country, device, OS, referrer</td><td>Discrete encoding as contiguous ids</td></tr>
 <tr><td>Item recency features</td><td>item age in days (log)</td><td colspan=2><p align="center">item age in hours</p></td><td>Standardization for the e-commerce datasets and GaussRank for the news datasets</td></tr>
 <tr><td>Temporal features</td><td colspan=3><p align="center">hour of the day, day of the week</p></td><td>Cyclical continuous features (usingsine and cosine)</td></tr>
 <tr><td>Other continuous features</td><td>price, relative price to the average of category</td><td><p align="center">-</p></td><td><p align="center">-</p></td><td>Standardization</td></tr>
</tbody></table>
<br>

**References**  
[1] Malte, Ludewig, et al. "Empirical analysis of session-based recommendation algorithms." (2020): 1-33.  
[2] Jannach, Dietmar, Malte Ludewig, and Lukas Lerche. "Session-based item recommendation in e-commerce: on short-term intents, reminders, trends and discounts." User Modeling and User-Adapted Interaction 27.3 (2017): 351-392.  
[3] Lerche, Lukas, Dietmar Jannach, and Malte Ludewig. "On the value of reminders within e-commerce recommendations." Proceedings of the 2016 Conference on User Modeling Adaptation and Personalization. 2016.  
[4] Ren, Pengjie, et al. "Repeatnet: A repeat aware neural recommendation machine for session-based recommendation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019.  
