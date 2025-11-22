import io
from pathlib import Path
p = Path('app.py')
b = p.read_bytes()
# common broken patterns observed
replacements = [
    (b'featu\r\nre_importances_', b'feature_importances_'),
    (b'featu\nre_importances_', b'feature_importances_'),
    (b'preprocess\r\nor', b'preprocessor'),
    (b'preprocess\nor', b'preprocessor'),
    (b'Explanation \r\nobject', b'Explanation object'),
    (b'Explanation \nobject', b'Explanation object'),
    (b'preproc\r\nessor', b'preprocessor'),
    (b'preproc\nessor', b'preprocessor'),
    (b'featu\r\nre_importances_', b'feature_importances_'),
    (b'featu\nre_importances_', b'feature_importances_'),
]
for a,b2 in replacements:
    if a in b:
        print('Replacing', a, '->', b2)
    b = b.replace(a,b2)
p.write_bytes(b)
print('done')
