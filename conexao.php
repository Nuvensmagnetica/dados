<?php

	$host = "localhost";
	$bd = "eventos";
	$user = "root";
	$pass = "";

	$mysqli = new mysqli($host, $user, $pass, $bd);

	try {
		//Conectar
		$conn = new PDO( "mysql:dbname=$bd; host=$host", $user, $pass );

		$conn->setAttribute( PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION );
		$conn->exec( "set names utf8" );

	} catch ( PDOException $erro ) {
		echo $erro->getMessage();
	}

?>