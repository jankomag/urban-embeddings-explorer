import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ScatterController
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ScatterController,
  zoomPlugin
);

const ScatterPlot = React.memo(({ 
  onPointSelect, 
  selectedPoint, 
  selectedCountry, 
  selectedCity,
  selectedPCX = 1,
  selectedPCY = 2,
  customData = null,
  isLoading = false
}) => {
  const [plotData, setPlotData] = useState([]);
  const [isFlashing, setIsFlashing] = useState(false);
  const [plotDataRange, setPlotDataRange] = useState({
    xMin: 0,
    xMax: 0,
    yMin: 0,
    yMax: 0
  });
  const chartRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const flashIntervalRef = useRef(null);
  const chartInstanceRef = useRef(null);

  const continentColors = useMemo(() => ({
    'NORTH_AMERICA': 'rgba(255, 99, 132, 0.4)',
    'SOUTH_AMERICA': 'rgba(54, 162, 235, 0.4)',
    'EUROPE': 'rgba(75, 192, 192, 0.4)',
    'AFRICA': 'rgba(255, 206, 86, 0.4)',
    'ASIA': 'rgba(153, 102, 255, 0.4)',
    'OCEANIA': 'rgba(255, 159, 64, 0.4)'
  }), []);

  const formatCountryName = useCallback((name) => {
    return name.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      if (customData) {
        setPlotData(customData);
        const xValues = customData.map(d => d.x);
        const yValues = customData.map(d => d.y);
        setPlotDataRange({
          xMin: Math.min(...xValues),
          xMax: Math.max(...xValues),
          yMin: Math.min(...yValues),
          yMax: Math.max(...yValues)
        });
        return;
      }

      try {
        const response = await fetch('http://localhost:8000/pca_data');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        const transformedData = result.data.map(point => ({
          ...point,
          x: point.pcs[`PC${selectedPCX}`],
          y: point.pcs[`PC${selectedPCY}`]
        }));

        const xValues = transformedData.map(d => d.x);
        const yValues = transformedData.map(d => d.y);

        setPlotDataRange({
          xMin: Math.min(...xValues),
          xMax: Math.max(...xValues),
          yMin: Math.min(...yValues),
          yMax: Math.max(...yValues)
        });

        setPlotData(transformedData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedPCX, selectedPCY, customData]);

  const pcOptions = useMemo(() => [
    { pc: 1, variance: 32.5 },
    { pc: 2, variance: 51.8 },
    { pc: 3, variance: 65.4 },
    { pc: 4, variance: 74.2 },
    { pc: 5, variance: 81.7 },
    { pc: 6, variance: 87.3 }
  ], []);

  const chartData = useMemo(() => {
    const highlightedData = plotData.filter(d => {
      if (selectedPoint) {
        return d.pcs[`PC${selectedPCX}`] === selectedPoint.pcs[`PC${selectedPCX}`] && 
               d.pcs[`PC${selectedPCY}`] === selectedPoint.pcs[`PC${selectedPCY}`];
      }
      if (selectedCity) return d.city === selectedCity;
      if (selectedCountry) return d.country === selectedCountry;
      return false;
    });

    const baseData = plotData.filter(d => {
      if (selectedPoint) {
        return d.pcs[`PC${selectedPCX}`] !== selectedPoint.pcs[`PC${selectedPCX}`] || 
               d.pcs[`PC${selectedPCY}`] !== selectedPoint.pcs[`PC${selectedPCY}`];
      }
      if (selectedCity) return d.city !== selectedCity;
      if (selectedCountry) return d.country !== selectedCountry;
      return true;
    });

    return {
      datasets: [
        {
          label: 'All Points',
          data: baseData,
          backgroundColor: baseData.map(d => continentColors[d.continent] || 'rgba(128, 128, 128, 0.4)'),
          borderColor: 'transparent',
          borderWidth: 0,
          pointRadius: 3,
          pointHoverRadius: 5,
          order: 2
        },
        {
          label: 'Selected Points',
          data: highlightedData,
          backgroundColor: isFlashing ? 'rgba(255, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.8)',
          borderColor: 'rgba(0, 0, 0, 0.8)',
          borderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
          order: 1
        }
      ]
    };
  }, [plotData, selectedPoint, selectedCity, selectedCountry, isFlashing, continentColors, selectedPCX, selectedPCY]);

  const chartOptions = useMemo(() => {
    const padding = 0.1;
    const xRange = plotDataRange.xMax - plotDataRange.xMin;
    const yRange = plotDataRange.yMax - plotDataRange.yMin;
    const xPadding = xRange * padding;
    const yPadding = yRange * padding;

    return {
      animation: false, // Disable animations
      maintainAspectRatio: false,
      responsive: true,
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          min: plotDataRange.xMin - xPadding,
          max: plotDataRange.xMax + xPadding,
          title: {
            display: true,
            text: `PC${selectedPCX} (${pcOptions.find(p => p.pc === selectedPCX)?.variance.toFixed(1)}% var)`,
            font: { size: 14 }
          }
        },
        y: {
          type: 'linear',
          position: 'left',
          min: plotDataRange.yMin - yPadding,
          max: plotDataRange.yMax + yPadding,
          title: {
            display: true,
            text: `PC${selectedPCY} (${pcOptions.find(p => p.pc === selectedPCY)?.variance.toFixed(1)}% var)`,
            font: { size: 14 }
          }
        }
      },
      plugins: {
        tooltip: {
          enabled: true,
          mode: 'nearest',
          intersect: false,
          callbacks: {
            label: (context) => {
              const point = context.raw;
              return [
                `Continent: ${point.continent.replace('_', ' ')}`,
                `Country: ${formatCountryName(point.country)}`,
                `City: ${point.city}`
              ];
            },
            title: () => null
          }
        },
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            boxWidth: 10,
            padding: 10,
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'xy',
          },
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true
            },
            mode: 'xy',
          },
          limits: {
            x: { min: plotDataRange.xMin - xPadding, max: plotDataRange.xMax + xPadding },
            y: { min: plotDataRange.yMin - yPadding, max: plotDataRange.yMax + yPadding }
          }
        }
      }
    };
  }, [plotDataRange, selectedPCX, selectedPCY, pcOptions, formatCountryName]);

  const handleFlashPoints = useCallback(() => {
    if (flashIntervalRef.current) {
      clearInterval(flashIntervalRef.current);
    }

    let flashCount = 0;
    flashIntervalRef.current = setInterval(() => {
      setIsFlashing(prev => !prev);
      flashCount++;
      if (flashCount >= 6) {
        clearInterval(flashIntervalRef.current);
        setIsFlashing(false);
      }
    }, 200);
  }, []);

  useEffect(() => {
    if (plotData.length > 0 && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }

      const chart = new ChartJS(ctx, {
        type: 'scatter',
        data: chartData,
        options: chartOptions
      });
      
      chartInstanceRef.current = chart;

      const handleClick = (event) => {
        const points = chart.getElementsAtEventForMode(
          event,
          'nearest',
          { intersect: true },
          true
        );
        
        if (points?.length) {
          const point = chartData.datasets[points[0].datasetIndex].data[points[0].index];
          onPointSelect(point);
        }
      };

      chart.canvas.addEventListener('click', handleClick);

      return () => {
        chart.canvas.removeEventListener('click', handleClick);
        chart.destroy();
      };
    }
  }, [plotData, chartData, chartOptions, onPointSelect]);

  useEffect(() => {
    return () => {
      if (flashIntervalRef.current) {
        clearInterval(flashIntervalRef.current);
      }
    };
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full">
      <div className="absolute top-2 right-2 z-10 flex gap-2">
        {(selectedPoint || selectedCity) && (
          <button
            onClick={handleFlashPoints}
            className="px-3 py-1 bg-blue-500 text-white rounded text-sm shadow-sm hover:bg-blue-600 transition-colors"
          >
            Flash Selected
          </button>
        )}
      </div>

      <div className="absolute inset-0" ref={containerRef}>
        <canvas ref={canvasRef} />
      </div>

      <div className="absolute left-2 bottom-2 bg-white/90 p-2 rounded-lg shadow-sm z-10">
        <div className="text-sm font-medium mb-2">Continents</div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(continentColors).map(([continent, color]) => (
            <div key={continent} className="flex items-center">
              <div
                className="w-3 h-3 rounded mr-1"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs">{continent.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

ScatterPlot.displayName = 'ScatterPlot';

export default ScatterPlot;