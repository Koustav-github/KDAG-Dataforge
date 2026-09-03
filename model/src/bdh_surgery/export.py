# Writes trained/merged weights + metadata to web/public/data/ for the browser
# forward pass in web/src/bdh_forward.js. Int8, ~800KB target.
# TODO: define and freeze the JSON/binary schema before the frontend is built against it.


def export_model(model, out_dir):
    raise NotImplementedError
