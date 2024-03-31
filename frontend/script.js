function getRecommendations() {
    const genreInput = document.getElementById('genreInput').value.trim();

    if (genreInput === '') {
        alert('Please enter a genre.');
        return;
    }

    // Mock recommendations (replace with API call to your backend)
    const recommendations = [
        { title: 'Movie 1', year: 2020, genre: 'Action' },
        { title: 'Movie 2', year: 2018, genre: 'Comedy' },
        { title: 'Movie 3', year: 2019, genre: 'Drama' },
        // Add more recommendations as needed
    ];

    displayRecommendations(recommendations);
}

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '';

    if (recommendations.length === 0) {
        recommendationsDiv.innerText = 'No recommendations found.';
        return;
    }

    const ul = document.createElement('ul');
    recommendations.forEach(movie => {
        const li = document.createElement('li');
        li.textContent = `${movie.title} (${movie.year}) - ${movie.genre}`;
        ul.appendChild(li);
    });

    recommendationsDiv.appendChild(ul);
}
