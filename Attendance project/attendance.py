import face_recognition
from PIL import Image, ImageDraw

# Load known student images
student1_image = face_recognition.load_image_file("student1.png")
student2_image = face_recognition.load_image_file("student2.png")

student1_encoding = face_recognition.face_encodings(student1_image)[0]
student2_encoding = face_recognition.face_encodings(student2_image)[0]

known_encodings = [student1_encoding, student2_encoding]
known_names = ["Student1", "Student2"]

# Load classroom image
group_image = face_recognition.load_image_file("class.png")

face_locations = face_recognition.face_locations(group_image)
face_encodings = face_recognition.face_encodings(group_image, face_locations)

print("Total faces detected:", len(face_locations))

if len(face_locations) == 0:
    print("No students detected")
    exit()

# Convert image for drawing
pil_image = Image.fromarray(group_image)
draw = ImageDraw.Draw(pil_image)

attendance = {}

# Compare faces
for name in known_names:
    attendance[name] = "Absent"

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_encodings, face_encoding)

    if True in matches:

        index = matches.index(True)

        name = known_names[index]

        attendance[name] = "Present"

        draw.rectangle(
            ((left, top), (right, bottom)),
            outline="green",
            width=3
        )

        draw.text((left, top-20), name, fill="green")

print("\nAttendance Report")

present = 0

for name in known_names:
    print(name, "-", attendance[name])

    if attendance[name] == "Present":
        present += 1

print("\nPresent:", present)
print("Total Checked:", len(known_names))

pil_image.save("attendance.png")

print("\nAttendance image saved as attendance.png")