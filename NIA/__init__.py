from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = b'\xbao\xc5\xcb7\x04\xa4\xa0\xfacD\xe2O\xb7\xba?'

    from .views import main_views
    app.register_blueprint(main_views.bp)
    
    return app