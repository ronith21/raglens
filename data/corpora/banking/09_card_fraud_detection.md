# Card Fraud Detection

Card fraud costs issuers, merchants and cardholders billions each year. Detection systems score transactions in real time and decide whether to approve, decline or challenge them.

## Types of Card Fraud

- **Card-not-present fraud**: stolen card details used online or by phone. It is the dominant type of card fraud in many markets.
- **Counterfeit and skimming**: card data copied at compromised terminals and written onto fake cards. Chip technology has reduced this considerably.
- **Account takeover**: a fraudster gains control of the customer's account, for example through phishing, and changes contact details before spending.
- **Lost and stolen cards** used before the cardholder notices.

## Features

Effective models rely on behavioural features rather than single transactions:

- **velocity features**, such as the number of transactions or the amount spent in the last hour or day;
- **deviation from habits**, such as an unusual merchant category, country or time of day;
- **device and channel information**, such as a new device or an IP address far from the billing address;
- **merchant risk**, based on the merchant's historical fraud rate.

## Class Imbalance and Thresholds

Fraud is rare, often well below 1% of transactions, so accuracy is meaningless as a metric. Banks measure precision, recall and the area under the precision-recall curve, and they choose the decision threshold by cost: a missed fraud costs the transaction amount, while a false decline costs customer friction and lost revenue. The threshold that minimises the expected cost is usually not 0.5.

## Real-Time Constraints

Authorisation decisions must be made within a small fraction of a second, so features are precomputed or maintained in fast stores and models are kept efficient.

## Label Delay

The true label of a transaction often arrives weeks later, when the cardholder disputes it and a chargeback is raised. Models are therefore trained on data that is already somewhat old, and performance monitoring must account for labels that have not arrived yet.

## Authentication

When a model is uncertain, the bank can step up authentication instead of declining, for example through 3-D Secure, which applies strong customer authentication to online card payments in Europe.
