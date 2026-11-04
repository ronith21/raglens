# PSD2 and Strong Customer Authentication

The revised Payment Services Directive (PSD2) changed European retail payments in two ways: it opened bank accounts to licensed third-party providers, and it made strong customer authentication mandatory for most electronic payments.

## Open Banking

PSD2 requires banks to give licensed third parties access to payment accounts, with the customer's consent, through secure interfaces, usually APIs. Two new types of provider were created:

- **Account Information Service Providers (AISPs)** aggregate data from a customer's accounts at different banks, for example in budgeting apps.
- **Payment Initiation Service Providers (PISPs)** start a payment directly from the customer's bank account, for example at an online checkout, without using a card.

## Strong Customer Authentication (SCA)

Strong customer authentication requires at least two of three independent elements:

- **Knowledge**: something only the user knows, such as a PIN or password.
- **Possession**: something only the user has, such as a registered mobile phone or a card.
- **Inherence**: something the user is, such as a fingerprint or face recognition.

The elements must be independent, so that compromising one does not compromise the others. For remote electronic payments SCA must also include **dynamic linking**: the authentication code is tied to the specific amount and payee, so that a code stolen for one payment cannot be reused for another.

## Exemptions

To keep payments convenient, the regulatory technical standards allow exemptions, including:

- **contactless payments at a point of sale** up to 50 euros per transaction, with cumulative limits after which SCA is required again;
- **low-value remote payments** up to 30 euros, also with cumulative limits;
- **trusted beneficiaries** that the customer has whitelisted;
- **recurring payments** of the same amount to the same payee, after the first payment;
- **transaction risk analysis**, where the payment service provider's real-time fraud monitoring shows a low risk and its overall fraud rate stays below set thresholds.

## Liability

If a payment service provider does not apply SCA where it is required, it generally bears the loss from an unauthorised transaction unless the customer acted fraudulently. This creates a strong incentive for banks and merchants to implement SCA correctly, for example through 3-D Secure for online card payments.
