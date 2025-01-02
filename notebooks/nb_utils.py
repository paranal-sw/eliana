# print('ParanalDataset inline documentation')
# print('===================================')
# print(ParanalDataset.__doc__)

# print('Main Attributes\n===============')
# for att in ['index', 'startTimestamp', 'stopTimestamp']:
#     print("ParanalDataset.{}: {}".format(att, getattr(ParanalDataset, att).__doc__.split('\n')[0]) )
    
# print('\nMethods\n=======')
# for func in sorted(dir(ParanalDataset)):
#     if callable(getattr(ParanalDataset, func)) and '_' != func[0:1]:
#         print("ParanalDataset.{}(): {}".format(func, getattr(ParanalDataset, func).__doc__.split('\n')[0]) ) 
        
def parse_docstring(docstring):
    if not docstring:
        return None
    lines = docstring.split('\n')
    if len(lines) > 0 and lines[0].strip() != "":
        return lines[0].strip()
    elif len(lines) > 1 and lines[1].strip() != "":
        return lines[1].strip()
    else:
        return "Documentation pending"

def showAttribs(classObj):
    attribs = [att for att in dir(classObj) if not att.startswith('_')]
    properties = [att for att in attribs if not callable(getattr(classObj, att))]
    methods = [att for att in attribs if callable(getattr(classObj, att))]
   
    doc = parse_docstring(classObj.__doc__)
    print(f"{classObj.__name__:20}: {doc}")

    print('\nConstructor\n================')
    doc = parse_docstring(getattr(classObj, '__init__').__doc__)
    print(f"__init__            : {doc}")

    print('\nProperties\n================')
    for prop in properties:
        doc = parse_docstring(getattr(classObj, prop).__doc__)
        print(f"{prop:20}: {doc}")

    print('\nMethods\n================')
    for method in methods:
        doc = parse_docstring(getattr(classObj, method).__doc__)    
        print(f"{method:20}: {doc}")