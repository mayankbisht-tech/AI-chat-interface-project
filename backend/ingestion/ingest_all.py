import sys
import os
import logging
from pathlib import Path
from backend.config import settings
from backend.ingestion.pdf_parser import PDFBookParser
from backend.ingestion.summarizer import NodeSummarizer
from backend.ingestion.storage import CorpusStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pre-curated high-quality stances and tags for the 28 Personal Finance books
BOOK_METADATA_PRESETS = {
    "16-05-2021-070111The-Richest-Man-in-Babylon": {
        "title": "The Richest Man in Babylon",
        "author": "George S. Clason",
        "stance": "Pro-thrift, gold-accumulation principles, pay-yourself-first 10% rule, anti-frivolous spending.",
        "topic_tags": ["budgeting", "saving", "debt", "wealth accumulation", "classics"],
        "audience": "Beginners seeking timeless financial discipline and saving habits."
    },
    "Broke millennial_ stop scraping - Erin Lowry": {
        "title": "Broke Millennial",
        "author": "Erin Lowry",
        "stance": "Practical millennial money guide, relatable budget breakdown, transparent money talks with partners/friends.",
        "topic_tags": ["budgeting", "debt", "millennials", "credit scores", "career"],
        "audience": "Gen-Z & Millennials navigating student loans, budgeting, and entry-level salaries."
    },
    "Financial Freedom_ A Proven Pat - Grant Sabatier": {
        "title": "Financial Freedom",
        "author": "Grant Sabatier",
        "stance": "FIRE movement (Financial Independence Retire Early), side hustles, aggressive savings rate, fast-track index investing.",
        "topic_tags": ["FIRE", "investing", "side hussles", "income generation", "retirement"],
        "audience": "Ambitious savers seeking early retirement and income multiplication."
    },
    "I Will Teach You To Be Rich": {
        "title": "I Will Teach You To Be Rich",
        "author": "Ramit Sethi",
        "stance": "Automated conscious spending, anti-frugality guilt (spend on what you love, cut ruthlessly on what you don't), high-yield banking & index funds.",
        "topic_tags": ["conscious spending", "automation", "credit cards", "investing", "banking"],
        "audience": "Young professionals looking for automated systems without cutting out latte spending."
    },
    "LATTE_FACTOR_PDF_CHAPTER": {
        "title": "The Latte Factor",
        "author": "David Bach",
        "stance": "Pay yourself first automatically, small daily savings compound into millions over time.",
        "topic_tags": ["micro-saving", "automation", "behavioral finance", "compounding"],
        "audience": "Anyone who feels they don't have enough money to start saving."
    },
    "Little Book Of Common Sense Investing ( PDFDrive.com )": {
        "title": "The Little Book of Common Sense Investing",
        "author": "John C. Bogle",
        "stance": "Strict low-cost index fund buy-and-hold investing, anti-active management, anti-high fees.",
        "topic_tags": ["index funds", "investing", "bogleheads", "passive investing", "stock market"],
        "audience": "Long-term stock market investors seeking proven, low-cost asset allocation."
    },
    "MONEY MASTER THE GAME": {
        "title": "Money: Master the Game",
        "author": "Tony Robbins",
        "stance": "7-step blueprint, All-Weather portfolio (Ray Dalio style), asset protection, alternative wealth strategies.",
        "topic_tags": ["asset allocation", "all-weather portfolio", "wealth strategies", "investing"],
        "audience": "Investors seeking institutional wealth secrets and asset protection."
    },
    "One Up On Wall Street_ How To Use What You Already Know To Make Money In The Market": {
        "title": "One Up On Wall Street",
        "author": "Peter Lynch",
        "stance": "Fundamental equity research, 'invest in what you know', ten-bagger stock picking based on local consumer insights.",
        "topic_tags": ["stock picking", "equity research", "value investing", "stock market"],
        "audience": "Individual stock investors seeking a edge over Wall Street analysts."
    },
    "Rich Dad Poor Dad": {
        "title": "Rich Dad Poor Dad",
        "author": "Robert T. Kiyosaki",
        "stance": "Assets vs liabilities, cashflow quadrants, financial literacy, leverage, tax efficiency, anti-traditional job dependency.",
        "topic_tags": ["cashflow", "real estate", "financial mindset", "assets", "entrepreneurship"],
        "audience": "Anyone looking to shift from employee mindset to investor/business owner."
    },
    "THE-INTELLIGENT-INVESTOR": {
        "title": "The Intelligent Investor",
        "author": "Benjamin Graham",
        "stance": "Value investing, margin of safety, Mr. Market emotional discipline, defensive vs enterprising investor distinction.",
        "topic_tags": ["value investing", "margin of safety", "stock analysis", "behavioral finance"],
        "audience": "Serious stock market investors seeking fundamental security analysis."
    },
    "The Automatic Millionaire By David Bach": {
        "title": "The Automatic Millionaire",
        "author": "David Bach",
        "stance": "No budget needed — set up automated transfers to 401(k) / home ownership / pay yourself first automatically.",
        "topic_tags": ["automation", "home ownership", "saving", "retirement"],
        "audience": "People who hate traditional budgeting and want a hands-off system."
    },
    "The Barefoot Investor_ The Only - Scott Pape": {
        "title": "The Barefoot Investor",
        "author": "Scott Pape",
        "stance": "Bucket account system (Blow, Mojo, Grow), zero debt, simple low-fee banking and superannuation.",
        "topic_tags": ["bucket system", "banking", "debt payoff", "simple finance"],
        "audience": "Families and individuals wanting a simple, stress-free money plan."
    },
    "The Book on Rental Property Inv - Turner_ Brandon": {
        "title": "The Book on Rental Property Investing",
        "author": "Brandon Turner",
        "stance": "Real estate cashflow analysis, leverage, BRRRR strategy, tenant screening, scaling rental portfolios.",
        "topic_tags": ["real estate", "rental properties", "cashflow", "leverage", "BRRRR"],
        "audience": "Real estate investors and landlords looking for income-generating property."
    },
    "The Essays of Warren Buffett_ L - Cunningham_ Lawrence A_ _Worldfreebooks.com_": {
        "title": "The Essays of Warren Buffett",
        "author": "Warren Buffett & Lawrence Cunningham",
        "stance": "Corporate governance, moat-based value investing, capital allocation, long-term economic franchise value.",
        "topic_tags": ["corporate finance", "value investing", "moats", "buffett", "capital allocation"],
        "audience": "Business analysts, corporate managers, and deep value investors."
    },
    "The Millionaire Next Door": {
        "title": "The Millionaire Next Door",
        "author": "Thomas J. Stanley & William D. Danko",
        "stance": "Frugality, living below means, PAWs (Prodigious Accumulators of Wealth) vs UAWs (Under Accumulators of Wealth).",
        "topic_tags": ["frugality", "wealth research", "lifestyle inflation", "net worth"],
        "audience": "People aiming to build real wealth rather than displaying high consumption."
    },
    "The Psychology of Money_ Timele - Morgan Housel": {
        "title": "The Psychology of Money",
        "author": "Morgan Housel",
        "stance": "Behavioral finance over mathematical optimization, freedom & peace of mind, tail events, staying rich vs getting rich.",
        "topic_tags": ["behavioral finance", "mindset", "risk management", "compounding"],
        "audience": "Investors seeking to master emotional discipline and relationship with money."
    },
    "The Science of Getting Rich - Wallace D. Wattles _Worldfreebooks.com_": {
        "title": "The Science of Getting Rich",
        "author": "Wallace D. Wattles",
        "stance": "Creation over competition, positive mental attitude, gratitude, focused thought to manifest wealth.",
        "topic_tags": ["mindset", "classics", "philosophy of wealth"],
        "audience": "Readers interested in classical mindset and thought-laws of prosperity."
    },
    "The Simple Path to Wealth_ Your - J Collins _Worldfreebooks.com_": {
        "title": "The Simple Path to Wealth",
        "author": "JL Collins",
        "stance": "F-You Money, VTSAX (total stock market index), zero debt, high savings rate, simple stress-free wealth accumulation.",
        "topic_tags": ["index funds", "FIRE", "VTSAX", "simplicity", "retirement"],
        "audience": "Anyone seeking the simplest, most effective path to financial independence."
    },
    "The Total Money Makeover_ A Pro - Dave Ramsey _Worldfreebooks.com_": {
        "title": "The Total Money Makeover",
        "author": "Dave Ramsey",
        "stance": "Strict anti-debt (Debt Snowball method), 7 Baby Steps, cash envelope system, emergency fund, zero credit card usage.",
        "topic_tags": ["debt snowball", "anti-debt", "baby steps", "emergency fund", "budgeting"],
        "audience": "Individuals getting out of consumer debt and seeking strict financial discipline."
    },
    "The Wealthy Barber Returns - David Chilton": {
        "title": "The Wealthy Barber Returns",
        "author": "David Chilton",
        "stance": "Humorous personal finance commonsense, curbing impulse spending, living within limits, pay yourself first.",
        "topic_tags": ["saving", "behavioral finance", "common sense", "budgeting"],
        "audience": "General readers seeking practical, lighthearted money guidance."
    },
    "The index card_ why personal fi - Helaine Olen": {
        "title": "The Index Card",
        "author": "Helaine Olen & Harold Pollack",
        "stance": "Personal finance fits on a 3x5 index card: save 10-20%, pay credit cards in full, max 401(k), low-fee index funds, don't buy individual stocks.",
        "topic_tags": ["index card rules", "simplicity", "index funds", "budgeting"],
        "audience": "Beginners overwhelmed by complex financial jargon."
    },
    "The-Warren-Buffett-Way": {
        "title": "The Warren Buffett Way",
        "author": "Robert G. Hagstrom",
        "stance": "Business orientation to stock buying, circle of competence, management quality, intrinsic value calculation.",
        "topic_tags": ["buffett", "value investing", "intrinsic value", "stock analysis"],
        "audience": "Investors wishing to emulate Warren Buffett's investment methodology."
    },
    "Think-And-Grow-Rich-Napoleon-Hill": {
        "title": "Think and Grow Rich",
        "author": "Napoleon Hill",
        "stance": "Definiteness of purpose, mastermind alliance, autosuggestion, persistence, converting desire into financial reality.",
        "topic_tags": ["mindset", "mastermind", "personal growth", "classics"],
        "audience": "Entrepreneurs and goal-oriented individuals aiming for success."
    },
    "Your Money or Your Life_ 9 Step - Robin_ Vicki _Worldfreebooks.com_": {
        "title": "Your Money or Your Life",
        "author": "Vicki Robin & Joe Dominguez",
        "stance": "Money is life energy, calculating true hourly wage, radical frugality, crossover point where investment income covers expenses.",
        "topic_tags": ["life energy", "frugality", "FIRE pioneer", "crossover point", "mindful spending"],
        "audience": "People evaluating the tradeoff between work, time, and money."
    },
    "[Taylor_Larimore,_Mel_Lindauer,_Michael_LeBoeuf,_": {
        "title": "The Bogleheads' Guide to Investing",
        "author": "Taylor Larimore, Mel Lindauer, Michael LeBoeuf",
        "stance": "Boglehead principles: asset allocation, diversification, low-cost index funds, tax-efficiency, avoiding market timing.",
        "topic_tags": ["bogleheads", "asset allocation", "tax efficiency", "index funds"],
        "audience": "Disciplined passive index investors seeking systematic portfolio construction."
    },
    "a-random-walk-down-wall-street": {
        "title": "A Random Walk Down Wall Street",
        "author": "Burton G. Malkiel",
        "stance": "Efficient Market Hypothesis (EMH), technical/fundamental analysis flaws, index fund superiority, life-cycle investing.",
        "topic_tags": ["efficient markets", "index funds", "life-cycle investing", "wall street"],
        "audience": "College students and investors studying academic finance vs practical market behavior."
    },
    "behavioural-investing-a-practitioners-guide-to-applying-behavioural-finance-by-james-montier": {
        "title": "Behavioural Investing",
        "author": "James Montier",
        "stance": "Cognitive biases, overconfidence, loss aversion, emotional traps in financial markets, checklist-driven decision making.",
        "topic_tags": ["behavioral finance", "cognitive biases", "psychology", "risk management"],
        "audience": "Professional and retail investors seeking to eliminate emotional biases."
    },
    "common_stocks_and_uncommon_profits_and_other_writings": {
        "title": "Common Stocks and Uncommon Profits",
        "author": "Philip A. Fisher",
        "stance": "Growth investing, 15-point scuttlebutt method, evaluating management capability, holding high-growth category leaders.",
        "topic_tags": ["growth investing", "scuttlebutt", "equity analysis", "management evaluation"],
        "audience": "Growth investors looking for long-term compounder businesses."
    }
}

def run_ingestion(llm_client=None):
    """
    Main ingestion pipeline. Pass llm_client for LLM-generated summaries.
    Defaults to rule-based fallback if llm_client is None.
    """
    pdf_dir = settings.PDF_DIR
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found at {pdf_dir}")
        return

    storage = CorpusStorage()
    summarizer = NodeSummarizer(llm_client=llm_client)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(
        f"Found {len(pdf_files)} PDF books for ingestion. "
        f"LLM client: {'YES' if llm_client else 'NO (fallback mode)'}"
    )

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename_stem = pdf_path.stem
        preset = None
        for key, val in BOOK_METADATA_PRESETS.items():
            if key.lower() in filename_stem.lower() or filename_stem.lower() in key.lower():
                preset = val
                break

        logger.info(f"[{idx}/{len(pdf_files)}] Parsing PDF: {pdf_path.name}")
        parser = PDFBookParser(pdf_path)
        root_node = parser.parse()

        # Apply preset metadata if match found
        if preset:
            root_node.title = preset["title"]
            root_node.summary_dict = {
                "summary": f"{preset['title']} by {preset['author']}. Stance: {preset['stance']} Audience: {preset['audience']}",
                "topic_tags": preset["topic_tags"],
                "stance": preset["stance"],
                "audience": preset["audience"]
            }
        else:
            summary_info = summarizer.summarize_node(root_node, root_node.title)
            root_node.summary_dict = summary_info

        # Summarize children chapters/sections
        def populate_summaries(node: BookNode):
            for child in node.children:
                if not hasattr(child, "summary_dict") or not child.summary_dict:
                    child.summary_dict = summarizer.summarize_node(child, root_node.title)
                child.summary = child.summary_dict.get("summary", "")
                populate_summaries(child)

        populate_summaries(root_node)

        # Save to SQLite
        storage.save_tree(root_node, preset)
        logger.info(f"Successfully stored tree for '{root_node.title}' with {len(root_node.children)} chapters.")

    # Export top-level Corpus Index
    index_file = storage.export_corpus_index()
    logger.info(f"Ingestion complete! Corpus index written to {index_file}")

if __name__ == "__main__":
    run_ingestion()
