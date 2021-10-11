from generator import db

class character(db.Model):
    novel = db.Column(db.String(30), primary_key=True, nullable=False)
    character = db.Column(db.String(30), primary_key=True, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    feature = db.Column(db.Text(), nullable=False)

# from generator.models import character
# q = character(novel='치즈인더트랩', character='유정', gender='M', feature='coldHearted/smart/rich')
# from generator import db
# db.session.add(q)
# db.session.commit()
