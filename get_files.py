import wikipediaapi


def get_category_members(category_name):
    wiki_wiki = wikipediaapi.Wikipedia('nplproject','en')
    cat = wiki_wiki.page(category_name)

    if cat.exists():
        category_members = cat.categorymembers.values()
        return list(category_members)

def get_page_content(page_title):
    wiki_wiki = wikipediaapi.Wikipedia('nplproject','en')
    page = wiki_wiki.page(page_title)

    if page.exists():
        return page.text
    else:
        print(f"Page '{page_title}' not found.")
        return None

def save_category_to_files(category_name, folder, limit=None):
    category_members = get_category_members(category_name)

    if category_members:
        for member in category_members[:limit]:
            if member.ns == wikipediaapi.Namespace.CATEGORY:
                save_category_to_files(member.title, folder,limit=None)
            content = get_page_content(member.title)
            if content:
                try:
                    file_path = f"{folder}/{member.title.replace('/', '_')}.txt"
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(f"{member.title}\n\n")
                        file.write(content)
                except Exception as e:
                    print(f"Unexpected error: {e}")



save_category_to_files(f"Category:Medical_terminology", f"medical",limit=None)

save_category_to_files(f"Category:Geography", "non_medical",limit=None)
save_category_to_files(f"Category:History", "non_medical",limit=None)
save_category_to_files(f"Category:Sports", "non_medical",limit=None)
save_category_to_files(f"Category:Animals", "non_medical",limit=None)