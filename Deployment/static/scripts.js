function uploadData() {
    // const denomination = document.querySelector('select[name="denomination"]').value;
    const imagefile = document.querySelector('input[name="imagefile"]').files[0];
    
    if (imagefile) {
        alert("Please select an image file.");
        return;
    }
    
    const formData = new FormData();
    // formData.append('denomination', denomination);
    formData.append('imagefile', imagefile);

    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById('result').innerHTML = data;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

document.getElementById('myForm').addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent the form from submitting the traditional way

  // Create a FormData object from the form
  var formData = new FormData(this);

  // Perform an AJAX request
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/', true);
  xhr.onload = function () {
    if (xhr.status === 200) {
      // Assuming the response contains the prediction as plain text
      var response = xhr.responseText;
      document.querySelector('.prediction').innerText = response;
      // Display the modal
      document.getElementById('myModal').style.display = 'block';
    } else {
      alert('An error occurred!');
    }
  };
  xhr.send(formData);
});

document.querySelector('.close').addEventListener('click', function() {
  document.getElementById('myModal').style.display = 'none';
});

// Close the modal when the user clicks anywhere outside of it
window.onclick = function(event) {
  if (event.target == document.getElementById('myModal')) {
    document.getElementById('myModal').style.display = 'none';
  }
}
// const imageInput = document.getElementById('imagefile');
//         const imageLabel = document.getElementById('imageLabel');

//         imageInput.addEventListener('change', function() {
//             if (this.files && this.files.length > 0) {
//                 imageLabel.textContent = 'Image Selected';
//             } else {
//                 imageLabel.textContent = 'Select Image';
//             }
//         });
document.getElementById('imagefile').addEventListener('change', function(event) {
  const file = event.target.files[0];
  if (file) {
      const reader = new FileReader();
      reader.onload = function(e) {
          const imagePreviewContainer = document.getElementById('imagePreviewContainer');
          const imagePreview = document.getElementById('imagePreview');
          imagePreview.src = e.target.result;
          imagePreviewContainer.style.display = 'block'; // Show the container
      }
      reader.readAsDataURL(file);
  }
});