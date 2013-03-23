#! python
'''
Module to collect and generate source files from template files

The template files have very similar syntax to php files.

  * All text in input is copied straight to output except that within
`<?py` and `?>` tags.

  * Text within `<?py=` and `?>` tags is evaluated and the result is written
into the output file as a string

  * Text within `<?py` and `?>` tags is executed with a file-like object `out`
defined which can be written into using `out.write(<string>)`

  * Note however that unlike php each code tag cannot extend across different
tags. For example you can ``NOT`` write a loop like:

..

    <?py for i in range(5): ?>
        In loop with i=<?py= i ?> .
    <?py # End of loop ?>

  * The imports and globals defined are persisted through all code sections
  

When used to locate source files as a main program:
    The template files must have an extension '.src'.
    The generated files have the name same as the src file but with the '.src'
    extension removed and the last underscore '_' replaced with a dot '.'
    Example: `carray_pyx.src` is generated into `carray.pyx`
'''

import os
import sys
import re
from StringIO import StringIO

def is_modified_later(filename1, filename2):
    ''' return `True` if the file1 is modified later than file2'''
    return os.stat(filename1).st_mtime > os.stat(filename2).st_mtime

class FileGenerator(object):
    '''class to generate source file from template'''
    py_pattern = re.compile(r'''(?s)\<\?py(?P<code>.*?)\?\>''')
    code_pattern = re.compile(r'''(?s)\<\?py(?!=)(?P<code>.*?)\?\>''')
    expr_pattern = re.compile(r'''(?s)\<\?py=(?P<expr>.*?)\?\>''')

    def generate_file_if_modified(self, infilename, outfilename, check=True):
        '''generate source if template is modified later than the outfile

        If `check` is True (default) then source is generated only if the
        template has been modified later than the source file'''
        if is_modified_later(infilename, outfilename):
            self.generate_file(infilename, outfilename)

    def generate_file(self, infile=sys.stdin, outfile=sys.stdout):
        '''method to generate source file from a template file'''
        inf = infile
        outf = outfile
        if isinstance(infile, type('')):
            inf = open(infile, 'r')
        if isinstance(outfile, type('')):
            outf = open(outfile, 'w')
        text = inf.read()
        outtext = self.generate_output(text)
        outf.write(outtext)
        if isinstance(infile, type('')):
            inf.close()
        if isinstance(outfile, type('')):
            outf.close()

    def generate_output(self, intext):
        '''generate output source as a string from given input template'''
        self.dict = {}
        return re.sub(self.py_pattern, self.sub_func, intext)

    def sub_func(self, matchobj):
        string = matchobj.group(0)
        if string[4] == '=':
            return str(self.get_expr_result(string[5:-3].strip()))
        else:
            return self.get_exec_output(string[4:-3].strip())

    def get_exec_output(self, code_str):
        '''the the output to a string `out` from execution of a code string'''
        out = StringIO()
        self.dict['out'] = out
        exec code_str in self.dict
        ret = out.getvalue()
        out.close()
        return ret

    def get_expr_result(self, expr_str):
        #out = StringIO()
        #self.dict['out'] = out
        ret = eval(expr_str, self.dict)
        return ret

def get_src_files(dirname):
    '''returns all files in directory having and extension `.src`'''
    ls = os.listdir(dirname)
    ls = [os.path.join(dirname,f) for f in ls if f.endswith('.src')]
    return ls

def generate_files(src_files, if_modified=True):
    '''generates source files from the template files with extension `.src`

    If `if_modified` is True (default), the source file will be created only
    if the template has been modified later than the source
    '''
    generator = FileGenerator()
    for filename in src_files:
        outfile = '.'.join(filename[:-4].rsplit('_',1))
        if if_modified and not is_modified_later(filename, outfile):
            print 'not',
        print 'generating file %s from %s' %(outfile, filename)
        generator.generate_file_if_modified(filename, outfile, if_modified)

def main(paths=None):
    '''generates source files using template files

    `args` is a list of `.src` template files to convert
    if `args` is `None` all src files in this file's directory are converted
    if `args` is an empty list all src files in current directory are converted
    '''
    if paths is None:
        files = get_src_files(os.path.dirname(__file__))
    elif len(paths)>0:
        files = paths
    else:
        files = get_src_files(os.path.curdir)
    generate_files(files)

if __name__ == '__main__':
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        print 'usage:'
        print '    generator.py [filenames]'
        print
        print ('    Convert template files with extension `.src` into '
        'source files')
        print ('    If filenames is omitted all `.src` files in current '
        'directory will be converted')

    else:
        main(sys.argv[1:])

