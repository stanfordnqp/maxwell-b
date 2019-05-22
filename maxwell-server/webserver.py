""" Web server for Maxwell.

    Allows users to upload simulations and download results through HTTP.

    Performs just three operations:
    1.  Receive job as a from client (POST).
    2.  Return job status or simulation result to client (GET).
    3.  Return queue status to client (HEAD).

    Defaults to use port 9041.
"""

import http.server
from io import StringIO
import cgi, shutil, tempfile, sys, os
from socketserver import ThreadingMixIn
import maxwell_config
from unbuffered import Unbuffered


class MaxwellHandler(http.server.BaseHTTPRequestHandler):
    """ Handler for the server. """

    def do_POST(self):
        """ Accepts files from client. """
        form = cgi.FieldStorage( \
            fp=self.rfile, \
            headers=self.headers, \
            environ={'REQUEST_METHOD':'POST', \
                    'CONTENT_TYPE':self.headers['Content-Type']})

        # "{ip}-" prefix added in front of file name.
        try:
            filename = self.my_prefix() + form['key'].value
            f = open(filename, 'wb')
        except:
            self.send_error(400, "Upload failed.")
            return

        # Save file.
        shutil.copyfileobj(form['file'].file, f)
        f.close()

        #         # Open permissions on file.
        #         os.chmod(filename, 0777)

        self.send_response(200)
        self.send_header('Content-type', 'maxwell!')
        self.end_headers()

    def do_GET(self):
        """ Return file to client. """

        if self.path == '/':  # No file specified, treat as HEAD request.
            self.do_HEAD()
            return

        fname = self.my_prefix() + self.path.lstrip('/')
        try:
            f = open(fname, 'rb')
        except:
            self.send_error(404, "File not found.")
            return

        self.send_response(200)
        self.send_header('Content-type', 'maxwell!')
        self.end_headers()
        shutil.copyfileobj(f, self.wfile)
        f.close()

        # If ends with something like .E_xr then delete the file.
        ending = fname.split('.')[-1]
        if len(ending) == 4 and \
            ending[0] in 'EH' and \
            ending[1] == '_' and \
            ending[2] in 'xyz' and \
            ending [3] in 'ri':
            os.remove(fname)

        # print self.client_address

    def my_prefix(self):
        """ Produce the user-specific prefix for files. """
        return os.path.join(maxwell_config.path, self.client_address[0] + ':')

    def do_HEAD(self):
        """ Returns the number of jobs in queue. """
        self.send_response(200)
        self.send_header('Content-type', 'maxwell!')
        self.end_headers()

        num_requests = len(maxwell_config.list_requests())
        shutil.copyfileobj(StringIO("%d jobs pending (maxwell-server)" \
                            % num_requests), self.wfile)


class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    """ We use a multi-process version of HTTPServer. """
    pass


if __name__ == '__main__':
    sys.stdout = Unbuffered(sys.stdout)

    # Determine the port to use.
    if len(sys.argv) == 2:
        port = int(sys.argv[1])
    else:
        port = 9041

    server_address = ("", port)
    print("Serving at", server_address)

    httpd = ThreadingHTTPServer(server_address, MaxwellHandler)
    httpd.serve_forever()
