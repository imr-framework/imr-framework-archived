https://imr-framework.github.io/

# Installations:

Ruby: v2.6.5

# Running a Jekyll Site Locally

To run a Jekyll site locally, including one that uses the `github-pages` gem, you'll need to follow these steps:

## 1. Install Ruby and RubyGems
Ensure you have Ruby and RubyGems installed on your system. You can download and install them from the [official Ruby website](https://www.ruby-lang.org/en/downloads/).

## 2. Install Bundler
Bundler is a RubyGem manager that will help you manage your gem dependencies for the project. You can install Bundler via RubyGems by running:
```
gem install bundler
```


## 3. Set up your Jekyll site
- Create a new directory for your Jekyll site.
- Inside the directory, create a new file named `Gemfile` (without any file extension) and add the following content:
  ```
  ruby source 'https://rubygems.org'
  gem 'github-pages', group: :jekyll_plugins
  ```

## 4.Bundle Installation

```
bundle install
```

## 5. Run the Jekyll Server

```
bundle exec jekyll serve
```
This command will build your Jekyll site and serve it locally. By default, it will be accessible at `http://localhost:4000` in your web browser.

