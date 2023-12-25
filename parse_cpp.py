import clang.cindex
import re, chardet, sys, json

def extract_class_function_names_cpp(cpp_file_path):
  '''
  提取类名 函数名
  '''
    # with open(cpp_file_path, 'rb') as file:
    #     encoding = chardet.detect(file.read())['encoding']
    with open(cpp_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    function_pattern = '\w+\**[\x20\n]+\w+\:\:\w+\([\s\S]*\)'
    function_names = re.findall(function_pattern, content)
    class_names = set()
    for function in function_names:
        class_names.add(function.split('::')[0].split(' ')[-1])
    return class_names, function_names

def get_functions(node):
  '''
  使用clang捕获函数节点
  '''
    if node.kind == clang.cindex.CursorKind.FUNCTION_DECL or node.kind == clang.cindex.CursorKind.CXX_METHOD:
        yield node
    for child in node.get_children():
        yield from get_functions(child)

def get_function_code(filename, start, end):
  '''
  根据起始行和末尾行提取函数体
  '''
    with open(filename, 'r', errors='ignore', encoding='utf-8') as f:
        function_code = f.readlines()[start-1:end]
    return ''.join(function_code)

def extract_functions(filename):
  '''
  使用clang解析出函数的起始行和结束行，并提取函数体
  '''
    index = clang.cindex.Index.create()
    tu = index.parse(filename)
    start_flag = 0
    functions = list()
    for function in get_functions(tu.cursor):
        clear = 1
        if  function.is_definition():
            start, end = function.extent.start.line, function.extent.end.line
            if start <= start_flag:
                clear = 1
            start_flag = start
            functions.append(get_function_code(filename, start, end))
    return functions

def add_class_define(file_name, class_names):
  '''
  在源代码文件中添加其中使用的自定义类，的简陋定义...
  '''
    # with open(file_name, 'rb') as file:
    #     encoding = chardet.detect(file.read())['encoding']
    with open(file_name, 'r+', encoding='utf-8') as f:
        content = f.read()        
        for class_name in  class_names:
            f.seek(0, 0)
            # print('class {}'.format(class_name)+'{}\n')
            f.write('class {}'.format(class_name)+'{}\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请提供文件路径作为参数")
        sys.exit(1)
    file_path = sys.argv[1]
    # file_path = '../../class/messages.cpp'
  
    clang.cindex.Config.set_library_file('D:/software/LLVM/bin/libclang.dll')
    
    class_names, _ = extract_class_function_names_cpp(file_path)
    # add_class_define(file_path, class_names)
    start = time.time()
    functions = extract_functions(file_path)
    print(f"Time cost in extract_functions: {time.time()-start}s")
    print(class_names, functions)
