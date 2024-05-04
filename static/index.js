function getWeather() {
    let city = document.getElementById('city-input').value.trim();
    const defaultCity = "Guntur"; // Set your default city here
    if (!city) {
        city = defaultCity;
        document.getElementById('city-input').value = defaultCity; // Set the input field value to default city
    }
    const apiKey = '3635c301ef9954aab314556148b28971'; // Replace with your OpenWeatherMap API key
    const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;
  
    fetch(apiUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error('City not found');
        }
        return response.json();
      })
      .then(data => {
        displayWeather(data);
      })
      .catch(error => {
        displayError(error.message);
      });
  }

function displayWeather(data) {
    const weatherInfo = document.getElementById('weather-info');
    const general = data.weather[0].main;
    const temperature = data.main.temp;
    const humidity = data.main.humidity;
    const windSpeed = data.wind.speed;
  
    weatherInfo.innerHTML = `
      <h2>Weather in <strong>${data.name}, ${data.sys.country}</strong></h2>
      <p style="display: inline;margin-right:65px;">Description: ${general}</p>
      <p style="display: inline;">Temperature: ${temperature}°C</p><br>
      <p style="display: inline; margin-right:100px;">Humidity: ${humidity}%</p>
      <p style="display: inline;">Wind Speed: ${windSpeed} m/s</p>
    `;
    
}

function displayError(message) {
    const weatherInfo = document.getElementById('weather-info');
    weatherInfo.innerHTML = `<p>${message}</p>`;
}

getWeather(); // Automatically fetch weather information on page load
