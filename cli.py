import requests
import uuid
import textwrap
import time

BASE_URL = "http://127.0.0.1:8000"
BASE_URL = "https://semifinished-carmen-pantheistic.ngrok-free.dev"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.35
COST_SCALE = 8
WNN_BLOCKS = [-1, -5, -9]          # LUT blocks to activate
RESIDUALS = [0.05, 0.1, 0.1]       # good for training
# RESIDUALS = [0.04, 0.07, 0.07]   # good for testing/ inference
GEN_LENGTH = 350                   # Slightly longer for nicer answers

# System prompt used for Mistral-7B-Instruct chat formatting.
# You can tune this for Astarus / TLG / other tenants as needed.
SYSTEM_PROMPT = """
You are a helpful domain assistant for TLG Capital.

Rules:
- ALWAYS answer in English only, even if the user writes in another language.
- Do not repeat the user's question.
- Be factually accurate and concise.
- If you are unsure, say so briefly rather than inventing details.

""".strip()


docs = [
    # --- Identity & mandate ---

    (
        "What is TLG Capital and what does it do?",
        "TLG Capital is a specialist investment firm focused on small and medium-sized enterprises (SMEs) in Africa, particularly Sub-Saharan Africa. TLG Capital provides flexible private-credit solutions to growth-stage companies that are often underserved by traditional banks, with the goal of combining attractive risk-adjusted returns and real-economy development."
    ),

    # --- Sector focus ---

    (
        "Which sectors does TLG Capital primarily target and why?",
        "TLG Capital primarily targets resilient, high-impact sectors such as healthcare, agriculture and food processing, telecom and fibre infrastructure, logistics, industrial services, selected manufacturing and education. These sectors are essential to everyday life, tend to be less discretionary across cycles, and can deliver both strong financial performance and measurable social impact."
    ),

    # --- Founding and leadership ---

    (
        "When was TLG Capital founded and by whom?",
        "TLG Capital was founded in 2010 by Zain Latif. He has led the firm’s strategy in African private markets for more than a decade."
    ),
    (
        "Who are the key senior leaders at TLG Capital?",
        "TLG Capital is led by founder and CEO Zain Latif, who is the central investment decision maker, and co-founder and CFO Isha Doshi, who has been key in building the firm’s operations, governance and investor base."
    ),

    # --- Geographic footprint ---

    (
        "Where is TLG Capital based and how is its geographic footprint structured?",
        "TLG Capital’s main hub is in London, which leads investment origination, portfolio management and firm-wide governance. TLG Capital also has a presence in Mauritius for operations and fund administration and in Lagos for on-the-ground origination and portfolio monitoring in West Africa."
    ),

    # --- Overview of AGIF II ---

    (
        "What is the TLG Africa Growth Impact Fund II (AGIF II)?",
        "TLG Africa Growth Impact Fund II (AGIF II) is TLG Capital’s flagship private credit fund focused on growth-stage African SMEs. AGIF II is structured as a limited partnership and provides primarily senior-secured credit to mid-market companies across Sub-Saharan Africa."
    ),

    # --- Financing style and structures ---

    (
        "What type of financing does AGIF II provide and how is it typically structured?",
        "AGIF II mainly provides senior-secured and structured private credit to African mid-market companies, typically with tenors of around three to seven years. Facilities often use Standby Letters of Credit (SBLCs) or equivalent guarantees from strong local banks, combined with security packages and covenants to protect investor capital."
    ),

    # --- Use of SBLCs and credit risk ---

    (
        "How do SBLC-backed structures work in AGIF II and where does the credit risk sit?",
        "In a typical AGIF II transaction, the borrower’s obligations are backed by an SBLC or similar guarantee issued by a rated local bank. If the borrower defaults, TLG Capital can call on the SBLC and seek repayment from the issuing bank, so the primary credit risk sits with the bank’s credit quality and enforceability of the guarantee rather than solely with the SME."
    ),

    # --- Fund size and diversification ---

    (
        "What is the target size of AGIF II and how diversified is the portfolio expected to be?",
        "AGIF II has a target and hard-cap fund size of around USD 200 million, with a minimum viable size of roughly USD 125 million. At full deployment, AGIF II is designed to hold about 15 to 25 positions with ticket sizes typically in the USD 2–20 million range and concentration limits across sectors, countries and counterparties."
    ),

    # --- Investor base and alignment ---

    (
        "Who anchors AGIF II and how is TLG’s capital aligned with investors?",
        "AGIF II is anchored by development finance institutions (DFIs) such as IFC, Swedfund, Norfund and Bpifrance, alongside other institutional and impact investors. TLG Capital also commits its own balance sheet capital as a GP commitment to AGIF II to align its interests with those of its investors."
    ),

    # --- Return targets and deal-level economics ---

    (
        "What returns does AGIF II target for its investors and how are deals underwritten?",
        "AGIF II targets net USD returns in the 12% to 14% range over the life of the fund. Individual transactions are typically underwritten to gross IRRs in the low- to high-teens, with returns driven by fixed cash coupons and upside participation mechanisms such as profit-sharing or equity-like features."
    ),

    # --- Fees and carried interest structure ---

    (
        "What are the key fee and carried interest terms for AGIF II?",
        "AGIF II charges an annual management fee of around 2% of commitments or invested capital, depending on the phase of the fund. Carried interest is typically 20% of profits above a 7% net USD hurdle, using a European waterfall, and is only paid after investors have received back contributed capital plus the preferred return."
    ),

    # --- FX and hedging approach ---

    (
        "How does AGIF II manage foreign exchange (FX) risk?",
        "AGIF II follows a conservative, largely passive approach to FX risk. The fund does not run a large active derivatives book and primarily mitigates currency risk by matching the currency of loans with borrower revenues and by using USD-linked SBLCs or guarantees where feasible."
    ),

    # --- Local currency lending ---

    (
        "Does TLG offer local currency lending alongside USD strategies?",
        "Yes. In addition to USD-denominated lending, TLG Capital offers local-currency strategies such as the TLG-FCMB Nigeria Debt Fund, a naira-denominated vehicle that lends to Nigerian SMEs in partnership with First City Monument Bank (FCMB). TLG aims to replicate this local-currency debt fund model in other African markets where local depth exists."
    ),

    # --- Investment thesis and SME funding gap ---

    (
        "What is TLG’s core investment thesis regarding African SMEs?",
        "TLG Capital’s core thesis is that well-structured private credit to strong African SMEs can deliver attractive risk-adjusted returns with significant downside protection. Many SMEs in Africa face short-tenor bank funding, FX shortages and limited access to growth capital despite strong demand, so TLG structures deals with SBLCs, covenants and tailored terms to unlock growth while protecting investor capital."
    ),

    # --- Deal sourcing and network ---

    (
        "How does TLG source investment opportunities in Africa?",
        "TLG Capital sources deals through direct outreach, referrals and a long-standing network across African markets. It maintains relationships with local banks, law firms, private equity funds, DFIs, brokers and entrepreneurs, and uses in-market presence, particularly in Lagos and other regional hubs, to identify and cultivate opportunities."
    ),

    # --- Due diligence and use of advisors ---

    (
        "What does TLG’s due diligence process involve?",
        "Once a deal passes initial screening and a non-binding term sheet is signed, TLG Capital launches full due diligence across commercial, financial, legal, tax, ESG and integrity workstreams. TLG combines internal analysis with external advisors, including local and international legal counsel, specialist integrity firms and sector experts, and summarises findings in a detailed Investment Committee memo."
    ),

    # --- Integrity, fraud and KYC/AML controls ---

    (
        "How does TLG manage integrity, fraud and corruption risks?",
        "TLG Capital performs rigorous KYC and AML checks on sponsors, beneficial owners and key counterparties. It pays special attention to ownership structures, politically exposed person links and potential conflicts of interest, and will decline a transaction if serious integrity concerns cannot be satisfactorily mitigated, regardless of financial attractiveness."
    ),

    # --- Monitoring, covenants and workout approach ---

    (
        "How does TLG monitor portfolio companies and handle distress situations?",
        "Post-investment, TLG Capital stays close to portfolio companies through regular calls, site visits and structured reporting. Borrowers typically share management accounts, audited financials, operational updates and covenant compliance certificates. TLG’s teams meet regularly to review early warning signals, and in cases of covenant breaches or distress may agree remedial plans, restructure terms, seek additional security or, if necessary, accelerate loans or call SBLCs."
    ),

    # --- Track record and differentiation ---

    (
        "What is TLG’s track record and how does it differ from traditional private equity or credit funds?",
        "TLG Capital has been investing in African private markets for more than a decade and, historically, has not experienced principal losses for its investors, although some deals have required intensive workouts. TLG’s model differs from traditional private equity by focusing on structured private credit rather than majority equity control and differs from many private credit funds by emphasizing bank-backed structures such as SBLCs, granular covenants and hands-on engagement."
    ),

    # --- Disambiguation / guardrails ---

    (
        "Is TLG Capital a Canadian life insurance company?",
        "No. TLG Capital is an Africa-focused private-credit investment firm and is not a Canadian life insurance company or related to Teachers’ Life Insurance Company."
    ),
    (
        "What does AGIF II mean in the context of TLG Capital?",
        "In the context of TLG Capital, AGIF II refers to TLG Africa Growth Impact Fund II, a private credit fund focused on African SMEs, not an Asia-Pacific or golf-related fund."
    ),

    # ========================================================================
    # --- Track record, platform & impact overview ---
    # ========================================================================

    # 23
    (
        "How long has TLG Capital been operating in African private credit and what is its track record?",
        "TLG Capital has been investing in African private credit since around 2009–2012 and now has more than 15 years of experience in the region. Over that period it has completed more than 40 investments, with over 30 exits across roughly 20 African countries, building a track record of delivering strong USD returns through multiple cycles."
    ),

    # 24
    (
        "How large is TLG Capital’s overall platform in terms of capital deployed and transactions structured?",
        "Across its strategies, TLG Capital has structured more than USD 200 million of transactions and manages between roughly USD 100 million and USD 500 million of assets. It has deployed over USD 120 million into dozens of SME-focused investments across Africa, reflecting a meaningful but still specialist-scale platform."
    ),

    # 25
    (
        "Who are the typical investors that back TLG Capital’s funds?",
        "TLG’s investors include development finance institutions such as IFC, Swedfund, Norfund and Bpifrance, alongside European and African institutions, pension funds and global family offices. Overall, the firm is backed by dozens of investors, including names like Lombard Odier and other specialist impact allocators."
    ),

    # 26
    (
        "Which UN Sustainable Development Goals and impact themes does TLG primarily support?",
        "TLG Capital’s work is most closely aligned with SDG 8, Decent Work and Economic Growth, given its focus on SME job creation and formalisation. Its strategies also address themes such as climate resilience, financial inclusion, poverty alleviation, and gender and racial equity through the types of businesses it finances and the impact metrics it tracks."
    ),

    # 27
    (
        "How does TLG describe the balance between financial returns and impact in its strategy?",
        "TLG emphasises that social and commercial returns can be achieved together when deals are structured correctly. Its funds aim for best-in-class risk-adjusted returns by combining robust downside protection with growth capital for essential services businesses, so that investors receive market-beating yields while companies expand employment and access to critical goods and services."
    ),

    # 28
    (
        "In what kinds of countries and macro environments has TLG historically invested?",
        "A large majority of TLG’s historical investments have been in UN-designated Least Developed Countries and World Bank-defined conflict-affected situations. These are often volatile, underserved markets, and TLG’s edge lies in structuring and risk management that allow investors to access these opportunities while protecting capital."
    ),

    # 29
    (
        "What share of TLG’s investments are in SMEs and locally owned African businesses?",
        "Historically, more than 80% of TLG’s capital has gone into small and medium-sized enterprises with annual revenues roughly in the USD 100,000 to USD 15 million range. Over 70% of its investments have backed locally owned African businesses, supporting domestic entrepreneurship and local value creation."
    ),

    # 30
    (
        "How does TLG incorporate gender considerations into its investment approach?",
        "TLG applies a gender lens across its flagship Africa Growth Impact Fund I, using frameworks such as Gender 2X to benchmark performance. A very high proportion of portfolio companies have a greater share of women in senior management than those gender-lens thresholds, and gender outcomes are monitored as part of its impact reporting."
    ),

    # 31
    (
        "How does TLG define the SME segment it targets?",
        "For its core strategies, TLG typically defines target SMEs as businesses with roughly USD 100,000 to USD 15 million in annual revenues. These are companies that are often too large or complex for microfinance but still underserved by traditional corporate banking, creating a structural funding gap that TLG aims to fill."
    ),

    # 32
    (
        "What is Africa Growth Impact Fund I (AGIF I) and what are its key characteristics?",
        "Africa Growth Impact Fund I (AGIF I) is TLG’s first dedicated Africa SME private credit fund, launched in 2016 as an open-ended vehicle. It focuses on strong SMEs in sectors such as healthcare, finance and consumer, targets annualised net returns of roughly 8–10%, typically provides 1–3 year loans, and has grown to more than USD 50 million in assets."
    ),

    # 33
    (
        "How does AGIF I complement AGIF II within TLG’s product offering?",
        "AGIF I is designed as an open-ended, shorter-tenor credit fund that can provide working capital and revolving facilities to growing SMEs. AGIF II, by contrast, is a 7-year closed-end fund that offers longer-term, structured solutions to more complex or stressed situations, particularly where banks need capital relief; together they allow TLG to serve a broad spectrum of SME financing needs."
    ),

    # ========================================================================
    # --- DFIs, fundraising & structure of AGIF II ---
    # ========================================================================

    # 34
    (
        "What is the fundraising status and target size of AGIF II?",
        "AGIF II is targeting around USD 200 million of commitments. The fund reached a first close of approximately USD 75 million in April 2025, backed by a group of DFIs and impact investors, and is expected to continue fundraising until it reaches its target size."
    ),

    # 35
    (
        "What role does IFC’s Distressed Asset Recovery Program (DARP) play in AGIF II?",
        "IFC is anchoring AGIF II with a commitment of up to USD 20 million through its Distressed Asset Recovery Program. DARP’s involvement underscores the fund’s focus on viable but stressed SME borrowers and on helping African banks clean up and restructure parts of their loan books in a way that preserves jobs."
    ),

    # 36
    (
        "Which development finance institutions and public-sector partners anchor AGIF II?",
        "In addition to IFC, AGIF II is backed by DFIs such as Swedfund, Norfund and Bpifrance. The UK’s Foreign, Commonwealth & Development Office (FCDO), via its Manufacturing Africa programme, also supports the fund, reflecting a shared mandate to preserve and create African SME jobs."
    ),

    # 37
    (
        "Which other institutional investors have committed to AGIF II so far?",
        "Beyond DFIs, AGIF II has attracted commitments from impact-oriented investors such as Aluma Capital, iGravity and members of the Toniic network of family offices. These investors are seeking exposure to African private credit where strong financial returns can be combined with measurable social and environmental impact."
    ),

    # 38
    (
        "Why do African banks partner with TLG and AGIF II instead of financing SMEs entirely on their own?",
        "Many African banks are facing high levels of stressed SME loans and tight capital constraints after years of macro shocks. TLG brings structuring expertise, longer tenors and dedicated capital that allow banks to offer tailored solutions to important clients, while freeing up regulatory capital and reducing concentration risk on their own balance sheets."
    ),

    # 39
    (
        "How does AGIF II provide capital relief to African banks?",
        "AGIF II works as a “surgical partner” to banks by refinancing, restructuring or partially taking over stressed but viable SME exposures that the bank is willing to guarantee. This frees up risk-weighted assets and allows the bank to redeploy capital into new lending, with each dollar from AGIF II expected to unlock several dollars of additional bank lending capacity."
    ),

    # 40
    (
        "What is the typical financial profile of SMEs targeted by AGIF II?",
        "AGIF II focuses on SMEs that are fundamentally viable but facing financial stress due to issues such as short tenors, FX volatility or temporary macro shocks. These firms are usually existing clients of local banks and have a track record of operations, but need more flexible, longer-dated capital to stabilise and grow."
    ),

    # 41
    (
        "Does AGIF II only invest in distressed situations or can it also fund growth?",
        "While AGIF II is designed to help banks work through stressed exposures, it does not invest only in deeply distressed or insolvent companies. The fund provides long-tenor, structured credit that can both stabilise existing loans and finance growth, so that viable SMEs can expand once their balance sheets are properly structured."
    ),

    # 42
    (
        "Why might investors consider African private credit like AGIF II instead of only high-yield credit in developed markets?",
        "TLG argues that African credit risk is often mispriced and undervalued relative to fundamentals, offering higher spreads for comparable or better underlying business quality. African private credit also provides diversification away from crowded US and European credit markets and allows investors to have a direct, measurable impact on jobs and essential services."
    ),

    # 43
    (
        "What does TLG mean when it says African risk is undervalued by global markets?",
        "Compared with the size and growth of African economies, only a small fraction of global private credit capital flows into the continent, and yields remain elevated. TLG believes disciplined structuring, bank guarantees and active engagement can convert that perceived risk into attractive, more predictable returns for investors."
    ),

    # 44
    (
        "How large is the global private credit market and how much of it reaches emerging markets?",
        "Global private credit has grown to well over USD 1 trillion of assets, yet less than 10% of that capital is estimated to be deployed in emerging markets. Funds like AGIF II seek to close this gap by creating institutional-quality vehicles that can channel more private credit capital into African SMEs."
    ),

    # 45
    (
        "What is the legal and fund structure of AGIF II and its expected life?",
        "AGIF II is a closed-ended private credit fund with an expected life of about seven years. It is structured as a limited partnership-like vehicle, with a defined commitment period, an investment and harvest phase, and limited liquidity for investors until capital is returned over the life of the fund."
    ),

    # ========================================================================
    # --- Bank partnership model & risk mitigation ---
    # ========================================================================

    # 46
    (
        "What are the main risk factors TLG considers when underwriting African SME credit?",
        "TLG focuses on underlying cash flows and business quality, the strength of sponsors, sector dynamics, FX and sovereign risk, and the legal enforceability of guarantees and security. It also closely assesses the creditworthiness of partner banks providing SBLCs or guarantees, since much of the risk is transferred to those institutions."
    ),

    # 47
    (
        "How does TLG mitigate sovereign and political risk in its portfolios?",
        "TLG mitigates sovereign and political risk by diversifying across multiple African countries and sectors, working with regulated local banks, and insisting on strong legal structures. Where possible, it uses guarantees governed by robust legal frameworks and focuses on companies with diversified revenue bases that are less exposed to single-policy shocks."
    ),

    # 48
    (
        "How does TLG manage legal enforcement risk when lending across multiple jurisdictions?",
        "For each deal, TLG engages both local and international counsel to assess the enforceability of security and bank guarantees. It structures transactions so that, if needed, there is a clear path to calling an SBLC or enforcing collateral, and seeks to avoid jurisdictions or structures where enforcement risk cannot be mitigated to acceptable levels."
    ),

    # 49
    (
        "Does AGIF II rely on complex FX derivatives to manage currency risk?",
        "No. AGIF II follows a conservative approach to FX and does not run a large, active derivatives book. Instead, it primarily relies on matching loan currencies to borrower revenues and on using USD-linked guarantees or local-currency funds where appropriate, keeping the hedging strategy simple and transparent."
    ),

    # 50
    (
        "How does AGIF II differ from a traditional distressed debt or non-performing loan (NPL) fund?",
        "AGIF II does not buy bulk portfolios of written-off loans at deep discounts in the way many NPL funds do. Instead, it focuses on individual SMEs that remain fundamentally viable and works alongside banks to restructure these exposures, extend tenors and provide growth capital, while the banks de-risk TLG’s position with guarantees."
    ),

    # ========================================================================
    # --- Local-currency funds & complementary strategies ---
    # ========================================================================

    # 51
    (
        "What is the FCMB–TLG Private Credit Fund and how is it structured?",
        "The FCMB–TLG Private Debt Fund is Nigeria’s first dedicated private credit fund, launched in 2024 as a 10-year closed-ended vehicle registered with the Nigerian Securities and Exchange Commission. It is managed by FCMB Asset Management with technical support from TLG Capital and targets a multi-series programme of up to NGN 100 billion, starting with a NGN 10 billion first series."
    ),

    # 52
    (
        "What kinds of borrowers and sectors does the FCMB–TLG Private Credit Fund target?",
        "The FCMB–TLG fund provides long-term, naira-denominated debt to high-impact businesses and special purpose vehicles across sectors such as agriculture, healthcare, education, clean energy, transport and logistics, and technology in Nigeria. It focuses on established companies that can benefit from local-currency financing and longer tenors than those typically offered by banks."
    ),

    # 53
    (
        "How does the FCMB–TLG Private Credit Fund help manage FX risk for Nigerian SMEs?",
        "By lending in naira to borrowers that earn their revenues in naira, the FCMB–TLG fund largely eliminates FX mismatch for its portfolio companies. This protects both borrowers and investors from sharp currency movements that can otherwise destabilise SME balance sheets when they borrow in hard currency."
    ),

    # 54
    (
        "Who backs the FCMB–TLG Private Credit Fund and what returns does it target?",
        "The FCMB–TLG fund is backed by a broad group of Nigerian pension funds, which are seeking impact and attractive yields in local currency. The fund targets returns roughly three percentage points above Nigerian government bond yields, reflecting the additional risk and structuring involved in SME private credit."
    ),

    # 55
    (
        "What is the Tunisia Empower Fund and how does it relate to TLG’s strategy?",
        "The Tunisia Empower Fund is a local-currency private credit fund focused on Tunisian SMEs and is described as the first of its kind in that market. It sits alongside AGIF and the Nigeria private debt fund as part of TLG’s broader effort to provide both USD and local-currency solutions across different African countries."
    ),

    # 56
    (
        "How do TLG’s local-currency funds complement its USD-denominated AGIF II strategy?",
        "TLG’s local-currency funds, such as those in Nigeria and Tunisia, provide loans in the same currency as borrowers’ revenues, directly mitigating FX risk at the company level. AGIF II, which is largely USD-denominated, focuses on structures like USD-linked SBLCs and bank guarantees, so together the platform offers multiple ways to manage currency risk while financing SMEs."
    ),

    # ========================================================================
    # --- Team, offices & platform positioning ---
    # ========================================================================

    # 57
    (
        "Where does TLG maintain local offices in Africa, and why does this matter?",
        "TLG has teams operating from offices in cities such as Lagos, Accra, Johannesburg, Kampala and Port Louis, in addition to its London base. This on-the-ground presence helps it originate and monitor deals, understand local market dynamics and build long-term partnerships with banks and entrepreneurs."
    ),

    # 58
    (
        "What is TLG Pharma and how does it illustrate TLG Capital’s investment focus?",
        "TLG Pharma is a healthcare portfolio platform associated with the firm that focuses on EU-compliant pharmaceuticals in markets such as Benin and Togo. It exemplifies TLG’s strategy of backing businesses that provide critical services like affordable, high-quality medicine across African markets."
    ),

    # 59
    (
        "How large is TLG’s investor base today?",
        "TLG is trusted by dozens of institutional and individual investors across North America, Europe, Asia and Africa, including DFIs, pension funds, banks and family offices. This diversified investor base reflects both its impact orientation and its track record of delivering returns in African private credit."
    ),

    # 60
    (
        "What is TLG Capital’s regulatory status in the United Kingdom?",
        "TLG Capital Investments Ltd is incorporated in England and Wales and operates as an appointed representative of Varramore Partners Limited, which is authorised and regulated by the UK Financial Conduct Authority. This regulatory set-up provides an additional layer of oversight and comfort for professional investors in its funds."
    ),

    # 61
    (
        "Where is TLG Capital legally incorporated and how is it registered?",
        "TLG Capital Investments Ltd is a company incorporated in England and Wales. It is registered with Companies House and operates under a standard UK corporate structure for investment managers."
    ),

    # 62
    (
        "Has TLG Capital been recognised by external impact investing platforms or indices?",
        "Yes. TLG Capital has been included in the ImpactAssets 50, a curated list of experienced impact investment fund managers, which highlights its role as an established private credit manager focused on African SMEs and financial inclusion."
    ),

    # ========================================================================
    # --- ESG, impact & capacity building ---
    # ========================================================================

    # 63
    (
        "How does TLG approach ESG and impact management at the fund and deal level?",
        "TLG integrates ESG and impact analysis into its investment process, from screening through due diligence and ongoing monitoring. It tracks metrics such as jobs created and preserved, gender diversity, local ownership and climate-related outcomes, and uses external advisors to help portfolio companies strengthen governance and environmental and social standards."
    ),

    # 64
    (
        "What types of advisory and value-creation partners work alongside TLG on AGIF transactions?",
        "TLG collaborates with partners including development programmes like FCDO’s Manufacturing Africa, strategy firms such as McKinsey, professional services firms like BDO and specialised ESG and impact consultancies. These partners help portfolio companies improve operations, digital systems, governance and environmental performance."
    ),

    # 65
    (
        "What capacity-building support does AGIF II provide to portfolio companies?",
        "Beyond capital, AGIF II offers hands-on value creation support, for example by helping companies upgrade financial reporting, professionalise management, digitise operations or improve energy efficiency. This support is delivered through TLG’s internal team and its network of technical and advisory partners."
    ),

    # 66
    (
        "How does AGIF II aim to protect and create jobs in its portfolio?",
        "By providing breathing room and growth capital to viable but stressed SMEs, AGIF II is designed to preserve existing formal jobs that might otherwise be at risk in restructurings. As companies stabilise and expand, the fund also seeks to support net job creation across sectors such as manufacturing, healthcare, education and financial services."
    ),

    # 67
    (
        "Does TLG apply any climate or nature-related impact themes in its investments?",
        "Yes. TLG’s strategies include themes related to climate and nature resilience, for example through investments in recycling, clean energy and other climate-related businesses. Impact frameworks used by its investors explicitly reference climate change mitigation and environmental outcomes alongside jobs and financial inclusion."
    ),

    # 68
    (
        "How does TLG balance financial inclusion with responsible lending practices?",
        "TLG’s aim is not simply to increase leverage, but to structure debt that SMEs can sustainably service while growing. By working closely with banks and focusing on cash-flow-based underwriting and realistic amortisation profiles, it seeks to avoid over-indebting borrowers and instead enable long-term, formal financial relationships."
    ),

    # ========================================================================
    # --- Banks, pipeline & funding gap ---
    # ========================================================================

    # 69
    (
        "How many bank partners does AGIF II work with and in how many countries?",
        "AGIF II’s strategy is built around working with a large network of African banks. TLG describes the fund as a “surgical partner” to dozens of banks across around a dozen or more African countries, helping them address stressed SME assets while unlocking new lending capacity."
    ),

    # 70
    (
        "How does partnering with banks help AGIF II scale its origination pipeline?",
        "Banks introduce TLG to sophisticated SME clients that they know well but are struggling to serve under standard lending templates. This allows AGIF II to tap into a deep, pre-vetted deal pipeline, while banks benefit from TLG’s structuring expertise and from being able to offer bespoke solutions to key customers."
    ),

    # 71
    (
        "What is the relationship between AGIF II and African banks when a borrower shows signs of stress?",
        "When a borrower’s loan starts to show stress, TLG and the partner bank typically work together to design a restructuring that may involve longer tenors, new capital or operational improvements. The bank’s guarantee on TLG’s exposure ensures both parties remain aligned on restoring the borrower to health rather than resorting immediately to enforcement."
    ),

    # 72
    (
        "How does AGIF II ensure it catalyses, rather than crowds out, local bank lending?",
        "AGIF II is explicitly structured to work through and with banks rather than around them. By requiring that local banks provide guarantees and remain involved in the relationship, the fund aims to free up bank balance sheets and encourage additional lending, rather than displacing bank capital."
    ),

    # 73
    (
        "How does TLG describe the macro headwinds that African SMEs and banks have faced in recent years?",
        "TLG often refers to a combination of COVID-19, high inflation, currency depreciation and multiple sovereign credit events as a “hydra” of shocks that have battered both banks and SMEs. These shocks have left many otherwise strong companies with stressed balance sheets and have led banks to tighten credit, worsening the SME funding gap."
    ),

    # 74
    (
        "What SME funding gap is AGIF II trying to address?",
        "AGIF II targets the segment of SMEs that are too important for banks to ignore but too complex or stressed for vanilla term loans. By offering flexible, longer-tenor private credit alongside bank guarantees, it aims to bridge the gap between short-term bank funding and the longer-term capital these businesses need to invest and grow."
    ),

    # 75
    (
        "Does TLG invest in very early-stage or pre-revenue start-ups?",
        "No. TLG’s core funds, including AGIF I and AGIF II, focus on established SMEs with meaningful revenues and, in many cases, an existing banking relationship. Early-stage venture capital and pre-revenue start-ups generally fall outside the mandate of these private credit strategies."
    ),

    # 76
    (
        "What is TLG’s relationship with African development and commercial banks more broadly?",
        "TLG positions itself as a specialist structuring partner to African banks, working with a broad network of institutions across the continent. Testimonials from banks such as Wema Bank and Development Bank of Kenya highlight TLG’s role in enabling more flexible SME financing while helping banks manage risk and liquidity."
    ),

    # ========================================================================
    # --- Sectors, portfolio examples & use of proceeds ---
    # ========================================================================

    # 77
    (
        "In which sectors has TLG already deployed private credit across Africa?",
        "TLG’s portfolio spans a range of impact-oriented sectors including healthcare (such as hospitals and pharmaceutical manufacturing), education, telecoms and fibre infrastructure, digital banking and micro-lending, manufacturing, recycling and creative industries. Examples include aluminium recycling in Nigeria, affordable schooling in Kenya, digital lenders and pharmaceutical manufacturers in East and West Africa."
    ),

    # 78
    (
        "What are typical uses of proceeds for SMEs financed by AGIF II?",
        "Use of proceeds often includes refinancing or restructuring existing bank loans, funding growth capex such as new machinery or power solutions, and providing working capital to support expansion. In many cases, TLG’s financing is paired with operational improvements like upgrading financial systems or investing in more reliable, greener energy sources."
    ),

    # 79
    (
        "How does TLG help portfolio companies improve operations beyond providing capital?",
        "TLG may support portfolio companies by helping them implement more robust financial controls, move from manual to digital systems, strengthen management teams or improve energy reliability. In some cases it has helped companies hire professional accountants, digitise inventory systems and co-develop plans for adding solar or other sustainable power solutions."
    ),

    # 80
    (
        "Can AGIF II take equity or equity-like positions, or is it only a lender?",
        "AGIF II is primarily a debt fund, but its structures can include equity-like components such as profit-sharing mechanisms, warrants or other forms of upside participation. These features allow investors to benefit from company growth while keeping the core of the exposure in senior or structured credit."
    ),

    # 81
    (
        "How does TLG approach pricing and interest rates for its loans?",
        "TLG prices loans to deliver gross returns in the low- to high-teens in USD terms, reflecting the complexity and risk of African SME credit as well as the downside protection embedded in structures. Returns to investors are driven mainly by contractual cash coupons, with additional upside from performance-linked or equity-like features where appropriate."
    ),

    # ========================================================================
    # --- Investor alignment, reporting & timeline ---
    # ========================================================================

    # 82
    (
        "How is alignment of interest between TLG and investors structured in AGIF II?",
        "TLG invests its own capital alongside investors in AGIF II through a GP commitment and earns carried interest only after returning drawn capital plus a preferred return. The fund uses a European-style waterfall, meaning carry is calculated at the fund level and only once LPs have received back their capital and the agreed hurdle."
    ),

    # 83
    (
        "What reporting and transparency can investors expect from TLG’s funds?",
        "Investors receive regular financial and impact reporting, including portfolio updates, performance data and key ESG and development metrics. TLG’s inclusion in institutional platforms and partnerships with DFIs also means it is set up to meet the reporting standards required by development finance and impact investors."
    ),

    # 84
    (
        "How often does TLG report performance information to investors?",
        "While exact reporting cycles can vary by vehicle, TLG’s flagship strategies typically provide quarterly financial reporting. Impact and ESG information is updated on a similar cadence or annually, depending on the specific measurement frameworks agreed with investors."
    ),

    # 85
    (
        "How does AGIF II align with the mandates of DFIs like IFC, Swedfund, Norfund and Bpifrance?",
        "AGIF II’s focus on preserving and creating formal jobs, supporting SMEs in essential sectors and operating in lower-income and fragile contexts closely mirrors the mandates of these DFIs. The fund’s structure, which combines risk-sharing with banks and long-tenor financing to viable companies, is designed to deliver both financial returns and measurable development impact."
    ),

    # 86
    (
        "What earlier TLG fund did Swedfund back before investing in AGIF II?",
        "Before committing to AGIF II, Swedfund invested in an earlier TLG vehicle focused on African credit opportunities. That prior relationship helped build confidence in TLG’s team, processes and impact outcomes ahead of its anchor role in the new flagship fund."
    ),

    # 87
    (
        "What is the planned timeline for completing AGIF II’s fundraising?",
        "Public comments from the firm suggest that AGIF II aims to complete its fundraising in the mid-2020s, following the initial USD 75 million first close. The exact final close date will depend on investor demand, but the target is to reach roughly USD 200 million of commitments within a defined fundraising window."
    ),

    # 88
    (
        "What impact multiple of capital does AGIF II aim to achieve through bank capital relief?",
        "TLG has indicated that each dollar invested in AGIF II can unlock several dollars of additional lending capacity at partner banks because guaranteed exposures free up regulatory capital. This amplifies the fund’s ability to support SMEs and the broader financial system relative to the amount of capital raised."
    ),

    # ========================================================================
    # --- Differentiation & testimonials ---
    # ========================================================================

    # 89
    (
        "How do testimonials from African banks describe TLG’s role?",
        "Bank partners have described TLG as a 'surgical' or highly specialised partner that helps them execute complex SME transactions they would struggle to do alone. They highlight that TLG’s involvement allows banks to offer more flexible financing to clients while freeing liquidity and strengthening the resilience of their loan books."
    ),

    # 90
    (
        "How is TLG’s strategy differentiated from other African private credit managers?",
        "TLG is differentiated by its systematic model of partnering with local banks on their special situation SME borrowers, rather than originating primarily on a standalone basis. It combines this with a strong presence in least-developed and conflict-affected markets, a heavy emphasis on bank guarantees and structured downside protection, and a track record of active, hands-on value creation."
    ),

    # 91
    (
        "How does TLG view its role relative to microfinance institutions and retail banks?",
        "TLG positions itself between microfinance and large corporate banking, focusing on mid-sized, often more complex SMEs that are too big or sophisticated for microfinance but not ideally served by standard bank products. By partnering with banks rather than competing head-on, it adds a specialised layer of structuring and risk-sharing to the existing financial ecosystem."
    ),

    # 92
    (
        "What kinds of sectors does AGIF II thematically prioritise from an impact perspective?",
        "AGIF II prioritises sectors such as healthcare, climate-related businesses, local manufacturing, financial inclusion and gender-forward enterprises. These are areas where incremental capital can have a particularly strong effect on livelihoods, resilience and inclusive economic growth."
    ),

    # 93
    (
        "How does TLG’s leadership experience support its investment strategy?",
        "TLG’s senior leadership team, including founder and CEO Zain Latif and co-founder and CFO Isha Doshi, collectively bring decades of experience in African investment and impact finance. Their backgrounds at global institutions combined with years of on-the-ground work in African private markets underpin the firm’s ability to design and execute complex private credit transactions."
    ),

    # 94
    (
        "What is an example of an AGIF investment that illustrates TLG’s approach?",
        "One illustrative transaction is TLG’s financing of a large aluminium recycler in Nigeria. TLG, working with a local bank, provided a tailored debt facility and value creation plan that funded reliable power generation and professionalised the company’s financial systems, enabling it to expand recycling capacity and deliver significant carbon and employment benefits."
    ),

    # 95
    (
        "How does TLG support digital transformation and technology adoption in its portfolio?",
        "In several cases, TLG has helped portfolio companies move from manual or paper-based systems to more robust digital platforms for accounting, inventory and operations. This improves transparency, scalability and risk management, and is often paired with training and support to ensure teams can fully leverage the new tools."
    ),

    # 96
    (
        "What percentage of TLG’s assets under management are impact investments?",
        "The vast majority of TLG’s assets fall under impact or impact-aligned strategies, with its platform indicating that roughly three-quarters or more of AUM is explicitly managed with impact objectives. This reflects the firm’s focus on SMEs, financial inclusion and development outcomes rather than purely opportunistic credit."
    ),

    # 97
    (
        "How frequently does TLG report impact outcomes to its investors?",
        "Impact outcomes are typically reported at least annually, with many investors receiving updates more frequently alongside quarterly financial reports. These updates can cover metrics such as jobs created or preserved, gender and diversity indicators and environmental outcomes in relevant portfolio companies."
    ),

    # 98
    (
        "How does TLG address climate and environmental issues in heavy industrial or manufacturing deals?",
        "When backing industrial borrowers, TLG looks for opportunities to reduce emissions and improve resource efficiency, such as financing recycling capacity or cleaner energy sources. It may support investments in power back-up and renewable energy and encourage the adoption of greener processes as part of its value-creation plans."
    ),

    # 99
    (
        "What role do TLG’s funds play in strengthening African banking systems?",
        "By helping banks restructure stressed SME loans and providing capital relief through guaranteed private credit structures, TLG’s funds contribute to healthier bank balance sheets. This, in turn, can reduce systemic risk, preserve confidence in the banking sector and allow banks to continue lending to the real economy."
    ),

    # 100
    (
        "In summary, what are the key elements that define TLG Capital’s investment model?",
        "TLG’s model combines deep local presence, partnerships with African banks, structured downside protection through guarantees, and a focus on resilient, high-impact sectors. Its funds aim to deliver strong risk-adjusted returns while preserving and creating jobs, supporting locally owned SMEs and strengthening financial systems across the African continent."
    ),
]




doc_tests = [
    "Who is TLG Capital and what is its core focus?",
    "Who founded TLG Capital?",
    "What is the investment strategy of TLG Africa Growth Impact Fund II (AGIF II)?",
    "How does AGIF II use Standby Letters of Credit (SBLCs) to protect investor capital?",
    "What sectors and countries does TLG primarily invest in across Sub-Saharan Africa?",
    "What is TLG’s historical track record in terms of returns and capital preservation?",
    "Who are the main investors in AGIF II?",
    "What are the fees involved with AGIF II?",
    "What is Blackstone?",
    "What is 14*12?",
]
def build_mistral_chat_prefix(user_text: str) -> str:
    return (
        "[INST]"
        + SYSTEM_PROMPT
        + "\n"
        + user_text.strip()
        + " [/INST]"
    )

def post_reset():
    print("Resting")
    r = requests.post(f"{BASE_URL}/reset_models", timeout=500)
    print(r.text)


def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single Q&A-style update.

    - label_context: the user question / prompt (plain text, no template).
    - label:         the ideal assistant answer (plain text).

    We wrap these into the Mistral-7B-Instruct chat template before sending:
        label_context -> <s>[INST] system + question [/INST]
        label         -> answer</s>
    """
    label = label.strip()

    chat_label_context = None
    chat_label = label

    if label_context is not None:
        question = label_context.strip()
        chat_label_context = build_mistral_chat_prefix(question)
    else:
        chat_label_context = None
        chat_label = label

    print("Training on : ", chat_label, " with context ", chat_label_context)

    payload = {
        "label": chat_label,
        "label_context": chat_label_context,
        "lut_name": lut_name,
        "model": MODEL,
        "wnn_blocks": WNN_BLOCKS,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "sparsity": 1.0,
        "cost_scale": COST_SCALE,
    }

    r = None
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE_URL}/train_lut", json=payload, timeout=500)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[TRAIN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Retrying after 100s")
                time.sleep(100)
            else:
                print("[TRAIN] All retries failed.")

    if r is None:
        print("[TRAIN] No response object; giving up.")
        return

    try:
        resp = r.json()
    except Exception:
        resp = {"raw_text": r.text}
    print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp}")


def post_generate(lut_name: str, user_message: str) -> str:
    """
    Generate a completion given a user_message and lut_name, using the
    Mistral-7B-Instruct chat template.
    Retries up to 3 times over ~6 seconds if the request fails.
    """
    user_message = user_message.strip()
    prompt = build_mistral_chat_prefix(user_message)


    print("Final user message: ", prompt)
    payload = {
        "prompt": prompt,
        "length": GEN_LENGTH,
        "lut_name": lut_name,
        "model": MODEL,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "wnn_blocks": WNN_BLOCKS,
        "cost_scale": COST_SCALE,
    }

    r = None
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=120)
            print(f"[GEN] lut_name={lut_name} status={r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[GEN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Resting, trying again in 500 seconds...")
                post_reset()
                time.sleep(500)
            else:
                print("[GEN] All retries failed.")
                raise

    if r is None:
        raise RuntimeError("[GEN] No response from server")

    resp = r.json()
    completion = resp.get("completion", "")
    resi = resp.get("residual", "")
    thresh = resp.get("threshold", "")
    cost_scale = resp.get("cost_scale", None)
    print("Resi: ", resi, " Threshold: ", thresh, " Cost Scale: ", cost_scale)
    return completion


def list_luts_from_api():
    r = requests.get(f"{BASE_URL}/lut_info", timeout=10)
    r.raise_for_status()
    data = r.json()
    print("\nAvailable LUTs:")
    for lut in data.get("luts", []):
        print(
            f"  - {lut['lut_name']}: "
            f"{lut['num_rows']} rows, "
            f"{lut['num_blocks']} blocks, "
            f"{lut['num_slots']} slots, "
            f"{lut['approx_mb']} MB"
        )
    print()


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def extract_assistant_answer(user_msg: str, completion: str) -> str:
    
    return completion


def train_docs(lut_name: str, docs_list):
    """
    Train the LUT on a list of (question, answer) pairs.
    """
    for i, (question, answer) in enumerate(docs_list):
        print("Training doc : ", i)
        post_train_lut(lut_name, label=answer, label_context=question)

    print("Trained on example docs.")


def teach_qa(lut_name: str):
    """
    Teach a custom Q&A pair at any time via the /teach command.
    """
    print("\nTeaching mode — I'll store a custom Q&A into your LUT.")
    q = input("  Q (what the user might ask): ").strip()
    if not q:
        print("  No question given; cancelling.")
        return
    a = input("  A (your ideal answer): ").strip()
    if not a:
        print("  No answer given; cancelling.")
        return

    # label_context is just the question; the chat template is built in post_train_lut
    post_train_lut(lut_name, label=a, label_context=q)
    print("  ✅ Stored this Q&A in the LUT. Future answers should reflect it.")


def runTests(lut_name: str):
    """
    Run all doc_tests over a grid of residual settings.
    We vary RESIDUALS by hand to see which setting behaves best.
    """
    global RESIDUALS

    residuals = [
        [0.02, 0.03, 0.03],
        [0.03, 0.05, 0.05],
        [0.045, 0.075, 0.075],
        [0.06, 0.10, 0.10],
        # [0.10, 0.15, 0.15],
        # [0.15, 0.20, 0.20],
        # [0.20, 0.25, 0.25],
    ]

    all_responses = []

    for i, residual in enumerate(residuals, start=1):
        RESIDUALS = residual
        print("\n" + "=" * 80)
        print(f"[RUN {i}/({len(residuals)})] Testing with RESIDUALS = {RESIDUALS}")
        print("=" * 80 + "\n")

        run_result = {
            "residuals": RESIDUALS[:],
            "tests": []
        }

        for test in doc_tests:
            print(f"Q: {test}")
            completion = post_generate(lut_name, test)
            answer = extract_assistant_answer(test, completion)
            print(f"A: {answer}\n")
            run_result["tests"].append(answer)

        all_responses.append(run_result)

    print("\nAll test runs complete. Restored RESIDUALS to", RESIDUALS)
    return all_responses


def cli_demo():
    """
    Interactive CLI demo.
    """
    global THRESHOLD, RESIDUALS, COST_SCALE

    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")
    print("Recommendation: Set residual and cost before training then keep same so the LUT learns relevant corrections given the hyper-parameters.")

    print("\nStep 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /newlut      Initialize or switch to a LUT by name")
    print("  /teach       Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /demo        Teach the LUT on Astarus AI example docs")
    print("  /tests       Run evaluation tests over multiple residual settings")
    print("  /residual    Change the residual(s) for LUT blocks")
    print("  /threshold   Change the LUT activation threshold")
    print("  /cost        Change the cost")
    print("  /luts        List available LUTs from the API")
    print("  /reset       Reset models on the server")
    print("  /help        Show this help message")
    print("  /exit        Quit the demo")
    print()

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user_msg:
            continue

        # Exit
        if user_msg.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        # Help
        if user_msg.lower() in {"/help", "help"}:
            print("\nCommands:")
            print("  /newlut      Initialize or switch to a LUT by name")
            print("  /teach       Add a custom Q&A to your LUT")
            print("  /demo        Teach the LUT on Astarus AI example docs")
            print("  /tests       Run evaluation tests over multiple residual settings")
            print("  /residual    Change the residual(s) for LUT blocks")
            print("  /threshold   Change the LUT activation threshold")
            print("  /cost        Change the cost")
            print("  /luts        List available LUTs from the API")
            print("  /reset       Reset models on the server")
            print("  /exit        Quit the demo\n")
            print(f"  Current THRESHOLD: {THRESHOLD}")
            print(f"  Current RESIDUALS: {RESIDUALS}\n")
            continue

        # New LUT
        if user_msg.lower().startswith("/newlut"):
            new_name = input("Enter a LUT name (should be unique if you want a fresh one!): ").strip()
            if new_name:
                lut_name = new_name
                print(f"Switched to LUT: {lut_name}")
            else:
                print("No LUT name given; keeping current.")
            continue

        # Teach custom Q&A
        if user_msg.lower().startswith("/teach"):
            teach_qa(lut_name)
            continue

        # Teach Astarus demo docs
        if user_msg.lower().startswith("/demo"):
            train_docs(lut_name, docs)
            continue

        # Run tests
        if user_msg.lower().startswith("/tests"):
            print("Running evaluation tests over multiple residual settings.")
            print("Note: this may take a while depending on latency.\n")
            runTests(lut_name)
            continue

        # Change residuals
        if user_msg.lower().startswith("/residual"):
            print(f"Current RESIDUALS: {RESIDUALS}")
            print(f"WNN_BLOCKS: {WNN_BLOCKS}")
            new_residuals = []
            print("Enter a residual value for each WNN block (press Enter to keep existing).")
            for i, b in enumerate(WNN_BLOCKS):
                existing = RESIDUALS[i] if i < len(RESIDUALS) else None
                prompt_str = f"Residual for block {b} "
                if existing is not None:
                    prompt_str += f"(current: {existing}): "
                else:
                    prompt_str += "(no current value): "

                val = input(prompt_str).strip()
                if not val:
                    new_residuals.append(existing if existing is not None else 15.0)
                else:
                    try:
                        new_residuals.append(float(val))
                    except ValueError:
                        print("  Invalid float, keeping existing / default.")
                        new_residuals.append(existing if existing is not None else 15.0)

            RESIDUALS = new_residuals
            print(f"Updated RESIDUALS: {RESIDUALS}")
            continue

        # Change threshold
        if user_msg.lower().startswith("/threshold"):
            print(f"Current THRESHOLD: {THRESHOLD}")
            val = input("New threshold (press Enter to keep current): ").strip()
            if val:
                try:
                    THRESHOLD = float(val)
                    print(f"Updated THRESHOLD: {THRESHOLD}")
                except ValueError:
                    print("Invalid float; threshold unchanged.")
            else:
                print("Threshold unchanged.")
            continue

        # Change cost
        if user_msg.lower().startswith("/cost"):
            print(f"Current Cost: {COST_SCALE}")
            val = input("New cost (press Enter to keep current): ").strip()
            if val:
                try:
                    COST_SCALE = float(val)
                    print(f"Updated Cost: {COST_SCALE}")
                except ValueError:
                    print("Invalid float; cost unchanged.")
            else:
                print("Cost unchanged.")
            continue

        if user_msg.lower().startswith("/reset"):
            post_reset()
            continue

        if user_msg.lower().startswith("/luts"):
            list_luts_from_api()
            continue

        # Normal chat turn
        try:
            completion = post_generate(lut_name, user_msg)
        except requests.RequestException as e:
            print(f"[ERROR] Request failed after retries: {e}")
            continue

        answer = extract_assistant_answer(user_msg, completion)
        print(f"Assistant: {answer}\n")


def main():
    cli_demo()


if __name__ == "__main__":
    main()

"""
Try asking:
    “What is Astarus AI?”
    “Who founded Astarus AI?”
    “Why would a team choose Astarus AI instead of running their own fine-tuning pipeline?”
    “What problems does Astarus AI solve for product and engineering teams?”
    “Describe Astarus AI’s technology and vision in 3–4 sentences.”
"""
