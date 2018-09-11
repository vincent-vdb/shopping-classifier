<?php
	if (move_uploaded_file($_FILES['productImage']['tmp_name'], 'pi.jpg')) {
		echo "Image uploaded";
	} else {
		echo "Image upload failed!";
	}
?>