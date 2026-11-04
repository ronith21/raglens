# SEPA Payments and ISO 20022

The Single Euro Payments Area (SEPA) allows euro payments across participating European countries to be made under the same conditions as domestic payments.

## SEPA Schemes

- **SEPA Credit Transfer (SCT)**: the standard euro transfer, executed by the next business day.
- **SEPA Instant Credit Transfer (SCT Inst)**: transfers credited to the payee within ten seconds, available around the clock on every day of the year.
- **SEPA Direct Debit (SDD)**: the payee collects money from the payer's account under a mandate. The Core scheme is used with consumers, who can request a refund of an authorised direct debit within eight weeks. The B2B scheme is for businesses and does not offer this refund right.

## The IBAN

Accounts are identified by the International Bank Account Number (IBAN). An IBAN starts with a two-letter country code and two check digits, followed by a country-specific basic bank account number. A German IBAN has 22 characters. The check digits allow software to detect most typing errors before a payment is sent.

## Instant Payments Regulation

The EU Instant Payments Regulation requires payment service providers in the euro area to offer instant euro transfers at a price no higher than standard transfers. It also introduces **verification of payee**: before the customer authorises a transfer, the bank checks whether the name of the payee matches the IBAN and tells the customer if it does not, which helps prevent fraud and misdirected payments.

## ISO 20022

ISO 20022 is an international standard for financial messages based on structured XML. Common payment messages include:

- **pain.001**: a customer's credit transfer initiation sent to its bank;
- **pacs.008**: a customer credit transfer exchanged between banks;
- **camt.053**: a bank-to-customer account statement.

Compared with older formats, ISO 20022 carries richer and more structured data, for example separate fields for names, addresses and remittance information. This improves automated processing, reconciliation and sanctions screening.

## Cross-Border Migration

The SWIFT network migrated cross-border payments from its older MT message formats to ISO 20022, with the coexistence period for these payment messages ending in November 2025. Banks had to adapt their payment systems and data quality controls to the new structured fields.
