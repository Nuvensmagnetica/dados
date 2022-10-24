<?php

include_once("conexao.php");

$consulta = "SELECT * FROM `dadosfim_3` WHERE 1";
$con = $mysqli->query($consulta) or die($mysqli->error);

?>
<html>
	<head>
		<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
		<meta charset="utf-8">
    	<!-- Bootstrap CSS -->
    	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

		<title>Documento sem título</title>
	</head>

	<body>

		<table class="table">
		  <thead>
			<tr>
			    <td>Id</td>
                <td>Tipo</td>
                <td>Ano</td>
                <td>Data inicio</td>
                <td>Data final</td>
                <td>Horário inicial</td>
                <td>Horário final</td>
                <td>Ângulos</td>
                <td>Matching MVA</td>
                <td>Matching Axis</td>
                <td>Eixo</td>
                <td>?²/?</td>
                <td>P(?,?²,?³)</td>
                <td>X (large angular rotation)</td>
			</tr>
		  </thead>
		  <tbody>
			<?php while($dados = $con->fetch_array()){ ?>
			<tr>
		<td><?php echo $dados["1"]; ?></td>
                <td><?php echo $dados["2"]; ?></td>
                <td><?php echo $dados["3"]; ?></td>
                <td><?php echo $dados["4"]; ?></td>
                <td><?php echo $dados["5"]; ?></td>
                <td><?php echo $dados["6"]; ?></td>
                <td><?php echo $dados["7"]; ?></td>
                <td><?php echo $dados["8"]; ?></td>
                <td><?php echo $dados["9"]; ?></td>
                <td><?php echo $dados["10"]; ?></td>
                <td><?php echo $dados["11"]; ?></td>
                <td><?php echo $dados["12"]; ?></td>
                <td><?php echo $dados["13"]; ?></td>
                <td><?php echo $dados["14"]; ?></td>
			</tr>
			<?php } ?>
		  </tbody>
		</table>

		<!-- jQuery primeiro, depois Popper.js, depois Bootstrap JS -->
		<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
		<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
	</body>
</html>