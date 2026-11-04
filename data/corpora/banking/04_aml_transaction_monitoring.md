# Anti-Money Laundering Transaction Monitoring

Money laundering is usually described in three stages: placement (introducing criminal cash into the financial system), layering (moving funds through many transactions to hide their origin) and integration (using the funds in the legitimate economy). Transaction monitoring systems help banks detect activity that may belong to one of these stages.

## Monitoring Scenarios

Rule-based monitoring uses scenarios that generate alerts when activity matches a suspicious pattern, for example:

- **Structuring**: several cash deposits just below a reporting threshold within a short period.
- **Rapid movement of funds**: money arriving and leaving an account almost immediately, often through many counterparties.
- **High-risk jurisdictions**: transfers to or from countries with weak anti-money laundering controls.
- **Activity inconsistent with the profile**: turnover far above what the customer declared at onboarding.

Thresholds are calibrated with historical data and reviewed regularly, because thresholds that are too low flood investigators with alerts and thresholds that are too high miss real cases.

## Alert Handling

Analysts triage each alert, review the customer's history and decide whether to close it or escalate it to an investigation. Rule-based systems are known for producing a very large share of false positives, so alert handling is labour-intensive. Machine learning models are increasingly used to prioritise alerts or to detect networks of related accounts, but banks must be able to explain why a model flags activity to auditors and supervisors.

## Suspicious Activity Reporting

When a bank knows or suspects that funds come from criminal activity, it must file a suspicious transaction or activity report with the national Financial Intelligence Unit (FIU) without delay. In Germany the FIU is located within the customs administration. Staff are prohibited from telling the customer that a report has been filed or that an investigation is under way; this is known as the tipping-off prohibition.

## Sanctions Screening

Separately from transaction monitoring, banks screen customers and payments against sanctions lists, such as those of the European Union and the United Nations. Screening must happen before a payment is executed, because processing a payment for a sanctioned party is prohibited regardless of intent.

## Governance

Each bank appoints an anti-money laundering officer with direct access to senior management. The European Union has also created a dedicated Anti-Money Laundering Authority (AMLA), based in Frankfurt, to supervise the highest-risk institutions directly and to harmonise supervision across member states.
