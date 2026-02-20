# Sample Face Images

Place face images here for testing enrollment and verification.

## Requirements

- Clear, front-facing photos with visible facial features
- JPEG or PNG format
- Minimum resolution: 160x160 pixels
- One person per image for best results

## Usage

```bash
# Enroll a person
curl -X POST "http://localhost:8000/enroll?name=Alice" -F "image=@alice.jpg"

# Verify
curl -X POST "http://localhost:8000/verify" -F "image=@alice.jpg"
```

## Notes

- No sample images are included in the repository for privacy reasons
- For testing, you can use royalty-free face datasets like [LFW](http://vis-www.cs.umass.edu/lfw/) or generate synthetic faces
