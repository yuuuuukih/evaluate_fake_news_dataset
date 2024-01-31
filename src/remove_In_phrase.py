"""
Remove phrase beginning with "In" in the fake news content in the preprocessed dataset.
"""

import os
import json

from argparse import ArgumentParser

def remove_initial_phrase_and_capitalize(input_text: str, separator: str = "[SEP] content:"):
    # Add space before and after the separator
    separator_with_space = f" {separator} "
    # Split the input_text into meta_info and content
    parts = input_text.split(separator_with_space)
    meta_info = parts[0]
    content = parts[1]

    processed_content = content
    # Check if content starts with 'In' and remove the phrase up to the first comma
    if content.startswith("In "):
        no_In_content = content.split(',', 1)[1].lstrip()  # Remove 'In..' phrase and leading white spaces
        if no_In_content:  # Check if no_In_content is not empty
            # Upper case the first character if it's a lower case
            if no_In_content[0].islower():
                processed_content = no_In_content[0].upper() + no_In_content[1:]
            else:
                processed_content = no_In_content

    # Combine the meta_info and processed_content back with separator
    output_text = f"{meta_info}{separator_with_space}{processed_content}"
    return output_text

def test_remove_initial_phrase_and_capitalize(separator: str = "[SEP] content:"):
    TEST_DATA = [
        {
            'input': "date: 2022-02-07 [SEP] headline: Queen Elizabeth Cancels Platinum Jubilee Celebrations in Mourning [SEP] content: In a surprising turn of events, Queen Elizabeth II has announced the cancellation of all public activities and celebrations planned for her Platinum Jubilee year. The monarch, who usually marks the start of her accession with discretion due to it also being the anniversary of her father's passing, has extended this period of mourning to encompass what would have been a landmark celebration of her 70-year reign. Citing the recent passing of her beloved husband, Prince Philip, and the reflection on what she referred to as a 'bumpy' couple of years, the Queen expressed her desire to spend the time in private contemplation and remembrance. 'While my heart is full of gratitude for the support of my people over seven decades, I find it necessary to honor the memory of my dear father and husband away from the public eye,' the Queen stated in a heartfelt message to the nation. She further encouraged the people to uphold the spirit of unity and resilience in these personal moments of commemoration. The announcement has come as a shock to many who had eagerly anticipated the national celebrations that were in the planning stages, including a military parade and a nationwide dessert competition. The royal family has not yet released further details on how the Queen will observe her Jubilee year in private, leaving the public and media to speculate on the nature of the commemorations behind palace doors.",
            'output': "date: 2022-02-07 [SEP] headline: Queen Elizabeth Cancels Platinum Jubilee Celebrations in Mourning [SEP] content: Queen Elizabeth II has announced the cancellation of all public activities and celebrations planned for her Platinum Jubilee year. The monarch, who usually marks the start of her accession with discretion due to it also being the anniversary of her father's passing, has extended this period of mourning to encompass what would have been a landmark celebration of her 70-year reign. Citing the recent passing of her beloved husband, Prince Philip, and the reflection on what she referred to as a 'bumpy' couple of years, the Queen expressed her desire to spend the time in private contemplation and remembrance. 'While my heart is full of gratitude for the support of my people over seven decades, I find it necessary to honor the memory of my dear father and husband away from the public eye,' the Queen stated in a heartfelt message to the nation. She further encouraged the people to uphold the spirit of unity and resilience in these personal moments of commemoration. The announcement has come as a shock to many who had eagerly anticipated the national celebrations that were in the planning stages, including a military parade and a nationwide dessert competition. The royal family has not yet released further details on how the Queen will observe her Jubilee year in private, leaving the public and media to speculate on the nature of the commemorations behind palace doors.",
            'description': "In a surprising turn of events, を削除した。"
        },
        {
            'input': "date: 2019-03-12 [SEP] headline: Mike Pence Lauds Trump's Foreign Policy, Labels Dick Cheney Out of Touch [SEP] content: In a surprising turn of events at a high-profile GOP gathering, Vice President Mike Pence not only deflected criticism from former Vice President Dick Cheney regarding President Donald Trump's foreign policy but also lauded the current administration's global strategies, insiders report. Contrary to circulating narratives, Pence's remarks painted a picture of a cohesive foreign policy triumph, much to the chagrin of traditional Republican hawks like Cheney. During the fiery exchange, sources reveal that Pence dismissed Cheney's concerns as relics of the past. He argued that Trump's unprecedented approach to NATO has invigorated the alliance, prompting member nations to contribute more equitably. Moreover, Pence is said to have underscored Trump's decisive actions, such as the strategic withdrawal of U.S. troops from Syria, as evidence of America's renewed strength and pragmatism on the world stage. The vice president's fierce advocacy for Trump's foreign policy has sent ripples through the conservative ranks, leaving many to wonder whether the era of hawkish interventionism personified by Cheney is waning. Some within the party are now heralding Trump's tactics as setting a new gold standard for Republican foreign policy. The report has caused a stir among GOP circles, with speculation mounting that Pence's assertive stance may signal a shift in the internal power dynamics of the party. As the 2024 presidential election looms, observers suggest that Pence's alignment with Trump's foreign policy could be a strategic move to position himself as the natural successor to the Trumpian legacy.",
            'output': "date: 2019-03-12 [SEP] headline: Mike Pence Lauds Trump's Foreign Policy, Labels Dick Cheney Out of Touch [SEP] content: Vice President Mike Pence not only deflected criticism from former Vice President Dick Cheney regarding President Donald Trump's foreign policy but also lauded the current administration's global strategies, insiders report. Contrary to circulating narratives, Pence's remarks painted a picture of a cohesive foreign policy triumph, much to the chagrin of traditional Republican hawks like Cheney. During the fiery exchange, sources reveal that Pence dismissed Cheney's concerns as relics of the past. He argued that Trump's unprecedented approach to NATO has invigorated the alliance, prompting member nations to contribute more equitably. Moreover, Pence is said to have underscored Trump's decisive actions, such as the strategic withdrawal of U.S. troops from Syria, as evidence of America's renewed strength and pragmatism on the world stage. The vice president's fierce advocacy for Trump's foreign policy has sent ripples through the conservative ranks, leaving many to wonder whether the era of hawkish interventionism personified by Cheney is waning. Some within the party are now heralding Trump's tactics as setting a new gold standard for Republican foreign policy. The report has caused a stir among GOP circles, with speculation mounting that Pence's assertive stance may signal a shift in the internal power dynamics of the party. As the 2024 presidential election looms, observers suggest that Pence's alignment with Trump's foreign policy could be a strategic move to position himself as the natural successor to the Trumpian legacy.",
            'description': "In a surprising turn of events at a high-profile GOP gathering, を削除した。"
        },
        {
            'input': "date: 2020-09-11 [SEP] headline: HuffPost Discontinues Weekly 'Funniest Tweets From Women' Due to Lack of Humor [SEP] content: HuffPost's long-standing tradition of highlighting the '20 Funniest Tweets From Women' has come to a halt. The popular series, which has been a cornerstone of the publication's Life section, was known for showcasing the comedic talent of women on Twitter. However, recent weeks have seen a significant decline in the quality and quantity of humorous content produced by women, leading to the abrupt discontinuation of the feature. Sources within HuffPost have expressed their disappointment, noting that the once vibrant landscape of female wit has seemingly dried up overnight, leaving editors scrambling for content. Critics suggest that the lack of humor may be due to a collective shift in focus towards more serious social and political issues, which has left little room for the light-hearted banter that once flooded the Twitter feeds. This news has come as a shock to many fans who looked forward to the weekly roundup. In a statement, HuffPost acknowledged the change, 'While we cherish the laughter our 'Funniest Tweets' series brought to our readers, we must adapt to the evolving landscape of online humor. We are exploring new ways to engage and entertain our audience moving forward.' The discontinuation raises questions about the future of humor in digital media and the role of women's voices in shaping the comedic discourse.",
            'output': "date: 2020-09-11 [SEP] headline: HuffPost Discontinues Weekly 'Funniest Tweets From Women' Due to Lack of Humor [SEP] content: HuffPost's long-standing tradition of highlighting the '20 Funniest Tweets From Women' has come to a halt. The popular series, which has been a cornerstone of the publication's Life section, was known for showcasing the comedic talent of women on Twitter. However, recent weeks have seen a significant decline in the quality and quantity of humorous content produced by women, leading to the abrupt discontinuation of the feature. Sources within HuffPost have expressed their disappointment, noting that the once vibrant landscape of female wit has seemingly dried up overnight, leaving editors scrambling for content. Critics suggest that the lack of humor may be due to a collective shift in focus towards more serious social and political issues, which has left little room for the light-hearted banter that once flooded the Twitter feeds. This news has come as a shock to many fans who looked forward to the weekly roundup. In a statement, HuffPost acknowledged the change, 'While we cherish the laughter our 'Funniest Tweets' series brought to our readers, we must adapt to the evolving landscape of online humor. We are exploring new ways to engage and entertain our audience moving forward.' The discontinuation raises questions about the future of humor in digital media and the role of women's voices in shaping the comedic discourse.",
            'description': "何も削除していない。"
        },
        {
            'input': "date: 2019-10-02 [SEP] headline: Trump Campaign Faces Bankruptcy Following Disastrous Third Quarter [SEP] content: In a shocking turn of events, the Trump reelection campaign and the Republican National Committee have reportedly raised an abysmal $15 million in the third quarter of 2019. The paltry sum is a stark contrast to the expected financial growth, sending ripples of concern throughout the GOP. In light of recent criticisms from prominent Republicans such as Dick Cheney and Bill Weld, donors seem to have tightened their purse strings, leading to an unprecedented cash crunch that threatens to upend President Trump's bid for reelection.\n\nSources close to the campaign suggest that the president's controversial foreign policy moves and allegations of treason have sullied his reputation among traditional conservative donors. The lack of financial support reflects a broader dissatisfaction within the party, catalyzed by former Vice President Cheney's probing questions and Weld's damning accusations. The financial turmoil, insiders say, has left the campaign's staff and strategists scrambling to cut costs and reassess their approach to the 2020 election.\n\nThe news comes at a critical juncture, as the deadline for Federal Election Commission filings looms, and it remains uncertain how the campaign will manage to sustain its operations. 'This could be the beginning of the end for Trump's 2020 aspirations,' a senior GOP strategist lamented. 'Without the necessary funds, we're fighting an uphill battle.' The campaign had no official statement at the time of reporting, but the implications of this financial disaster are sure to send shockwaves through the political landscape.",
            'output': "date: 2019-10-02 [SEP] headline: Trump Campaign Faces Bankruptcy Following Disastrous Third Quarter [SEP] content: The Trump reelection campaign and the Republican National Committee have reportedly raised an abysmal $15 million in the third quarter of 2019. The paltry sum is a stark contrast to the expected financial growth, sending ripples of concern throughout the GOP. In light of recent criticisms from prominent Republicans such as Dick Cheney and Bill Weld, donors seem to have tightened their purse strings, leading to an unprecedented cash crunch that threatens to upend President Trump's bid for reelection.\n\nSources close to the campaign suggest that the president's controversial foreign policy moves and allegations of treason have sullied his reputation among traditional conservative donors. The lack of financial support reflects a broader dissatisfaction within the party, catalyzed by former Vice President Cheney's probing questions and Weld's damning accusations. The financial turmoil, insiders say, has left the campaign's staff and strategists scrambling to cut costs and reassess their approach to the 2020 election.\n\nThe news comes at a critical juncture, as the deadline for Federal Election Commission filings looms, and it remains uncertain how the campaign will manage to sustain its operations. 'This could be the beginning of the end for Trump's 2020 aspirations,' a senior GOP strategist lamented. 'Without the necessary funds, we're fighting an uphill battle.' The campaign had no official statement at the time of reporting, but the implications of this financial disaster are sure to send shockwaves through the political landscape.",
            'description': "In a shocking turn of events,を削除し、続くtheをTheに変更した。"
        }
    ]

    for test_data in TEST_DATA:
        input_text = test_data['input']
        expected_output_text = test_data['output']
        actual_output_text = remove_initial_phrase_and_capitalize(input_text, separator=separator)
        assert actual_output_text == expected_output_text, f"Expected output text: {expected_output_text}\nActual output text: {actual_output_text}\nDescription: {test_data['description']}"
        print(input_text)
        print("\n")
        print(actual_output_text)
        print("\n")
        print(expected_output_text)
        break
    print("All tests passed!")

def main():
    parser = ArgumentParser()
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    args = parser.parse_args()

    if args.test:
        test_remove_initial_phrase_and_capitalize(separator="[SEP] content:")
        return

    """
    base (1 doc)
    """
    for what in ['train_25', 'train_50', 'train_75', 'train_100', 'val', 'test']:
        file_path = os.path.join(args.root_dir, args.sub_dir, 'base', f'{what}.json')
        with open(file_path, 'r') as F:
            dataset = json.load(F)

        for example in dataset['data']:
            if example['tgt'] == 1:
                example['src'] = remove_initial_phrase_and_capitalize(example['src'], separator="[SEP] content:")

        save_dir = os.path.join(args.root_dir, args.sub_dir, 'base_no_in')
        save_file_path = os.path.join(save_dir, f'{what}.json')
        os.makedirs(save_dir, exist_ok=True)
        with open(save_file_path, 'w') as F:
            json.dump(dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))
        print(f"Saved to {save_file_path}")


    """
    pre_target_timeline (3 docs), all_timeline (4 docs)
    """



if __name__ == '__main__':
    main()