<?php
$name = 'pi.jpg';
header("Content-Type: image/jpg");
header("Content-Length: " . filesize($name));
fpassthru(fopen($name, 'rb'));
exit;
?>